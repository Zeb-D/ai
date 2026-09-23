"""📱 网页版入口：手机浏览器输入 URL 就能聊天，**能力与控制台版一致**。

- 用的是同一个 [`core.Agent`](agent/core.py:143)：同一份人设（agent.md）、同一套「命令: / 完成:」协议、
  同一套技能文档与 `os.popen` 命令执行能力；
- 唯一区别是输出通道：控制台打印事件，这里把事件转成 SSE 推给浏览器（打字机效果）；
- 前端只有聊天：消息列表 + 输入框 + 发送/停止 + 「朗读回复」开关。

接口只有两个：
    GET  /            → 聊天页面（static/index.html，含 style.css / app.js）
    POST /api/chat    → SSE：`{"delta": ...}` 逐块文本，最后 `{"done": true, "answer": ...}`

启动：
    python3 agent/web.py                 # 默认 0.0.0.0:8000
    python3 agent/web.py --port 8899     # 换端口
    python3 agent/web.py --no-tools      # 关掉命令执行能力（只聊天）
"""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from config import (
    AGENT_FILE,
    STATIC_DIR,
    WEB_ENABLE_TOOLS,
    WEB_EXTRA_INSTRUCTIONS,
    WEB_HOST,
    WEB_MAX_HISTORY_MESSAGES,
    WEB_MAX_MESSAGE_CHARS,
    WEB_MAX_SESSIONS,
    WEB_PORT,
)
from core import Agent, Event, load_agent_config
from llm import DeepSeekLLM, LLMError

# ------------------------------------------------------------------ 单例（Agent 无状态，会话存在下面）
_LLM: DeepSeekLLM | None = None
_AGENT: Agent | None = None
_AGENT_LOCK = threading.Lock()


def get_agent(enable_tools: bool = WEB_ENABLE_TOOLS) -> Agent:
    """懒加载单例：控制台版与网页版是同一个 Agent 实现，这里只是换个输出通道。"""
    global _LLM, _AGENT
    with _AGENT_LOCK:
        if _AGENT is None:
            _LLM = DeepSeekLLM()
            _AGENT = Agent(
                _LLM,
                load_agent_config(AGENT_FILE),
                enable_tools=enable_tools,
                extra_instructions=WEB_EXTRA_INSTRUCTIONS,
            )
        return _AGENT


# ------------------------------------------------------------------ 会话（内存）
SESSIONS: dict[str, list[dict[str, str]]] = {}
SESSIONS_LOCK = threading.Lock()


def get_history(session_id: str) -> list[dict[str, str]]:
    with SESSIONS_LOCK:
        history = SESSIONS.get(session_id)
        if history is None:
            if len(SESSIONS) >= WEB_MAX_SESSIONS:  # 淘汰最早的会话，避免内存无上限
                SESSIONS.pop(next(iter(SESSIONS)), None)
            history = SESSIONS[session_id] = []
        return history


# ------------------------------------------------------------------ HTTP
class ChatHandler(BaseHTTPRequestHandler):
    server_version = "agent-webchat/1.0"
    protocol_version = "HTTP/1.0"  # 响应结束即关连接，SSE 天然可用

    def log_message(self, fmt: str, *args: Any) -> None:
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {fmt % args}", flush=True)

    # ------------------------------ 基础
    def _send_bytes(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        self._send_bytes(
            status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8"
        )

    def _read_json(self) -> dict[str, Any]:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError as exc:
            raise ValueError("Content-Length 不合法") from exc
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            data = json.loads(raw.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("请求体不是合法 JSON") from exc
        if not isinstance(data, dict):
            raise ValueError("请求体必须是 JSON 对象")
        return data

    # ------------------------------ SSE
    def _start_sse(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-transform")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")  # 反代不要缓冲
        self.end_headers()
        self.close_connection = True

    def _sse(self, payload: dict[str, Any]) -> None:
        self.wfile.write(f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8"))
        self.wfile.flush()

    # ------------------------------ 路由
    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler 约定
        path = urlparse(self.path).path
        try:
            if path.startswith("/api/"):
                self._send_json(404, {"error": "未知接口"})
            else:
                self._serve_static(path)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path != "/api/chat":
            self._send_json(404, {"error": "未知接口"})
            return
        try:
            self._handle_chat()
        except ValueError as exc:  # 参数问题：此时还没发响应头，可以正常回错误
            self._send_json(400, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception as exc:  # noqa: BLE001 - 兜底，别让线程崩掉
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    # ------------------------------ 静态文件
    def _serve_static(self, path: str) -> None:
        import mimetypes

        # 支持两种写法：/static/app.js（页面里的引用）与 /style.css（直接访问）
        relative = path[len("/static/"):] if path.startswith("/static/") else path
        relative = relative.lstrip("/") or "index.html"
        target = (STATIC_DIR / relative).resolve()
        if not target.is_relative_to(STATIC_DIR.resolve()) or not target.is_file():
            self._send_json(404, {"error": "文件不存在"})
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type == "application/javascript":
            content_type = f"{content_type}; charset=utf-8"
        self._send_bytes(200, target.read_bytes(), content_type)

    # ------------------------------ 聊天（与控制台版同一个 Agent 循环）
    def _emitter(self):
        """把 Agent 事件转成 SSE：模型文本 + 命令执行过程，最终回答在 done 里单独给。"""

        def emit(event: Event) -> None:
            if event.kind == "text":
                self._sse({"delta": event.text})
            elif event.kind == "command":
                self._sse({"delta": f"\n$ {event.text}\n"})
            elif event.kind == "result":
                self._sse({"delta": f"{event.text}\n"})
            elif event.kind == "notice":
                self._sse({"delta": f"\n（{event.text}）\n"})
            elif event.kind == "error":
                self._sse({"error": event.text})

        return emit

    def _handle_chat(self) -> None:
        payload = self._read_json()
        message = str(payload.get("message") or "").strip()
        if not message:
            raise ValueError("message 不能为空")
        if len(message) > WEB_MAX_MESSAGE_CHARS:
            raise ValueError(f"消息过长（上限 {WEB_MAX_MESSAGE_CHARS} 字）")
        session_id = str(payload.get("session_id") or "default")[:64]

        agent = get_agent()
        history = get_history(session_id)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": agent.system_prompt},
            *history[-WEB_MAX_HISTORY_MESSAGES:],
            {"role": "user", "content": message},
        ]

        self._start_sse()
        answer = ""
        try:
            answer = agent.run(messages, self._emitter())
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True

        # 把本轮（含命令与执行结果）同步回会话历史，保证多轮上下文一致
        with SESSIONS_LOCK:
            history[:] = messages[1:]
            del history[:-WEB_MAX_HISTORY_MESSAGES]

        try:
            self._sse({"done": True, "answer": answer})
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True


# ------------------------------------------------------------------ 启动
def lan_ipv4_addresses() -> list[str]:
    """枚举本机局域网 IPv4，用来提示「手机该输入哪个 URL」。"""
    addresses: set[str] = set()
    try:
        # 建立 UDP「连接」不会真的发包，只为拿到默认出口网卡的 IP
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            addresses.add(probe.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            addresses.add(info[4][0])
    except OSError:
        pass
    return sorted(ip for ip in addresses if not ip.startswith("127."))


def print_banner(host: str, port: int, agent: Agent) -> None:
    print(
        r"""
╭──────────────────────────────────────────────╮
│   📱 手机聊天网页（DeepSeek + 技能/命令）      │
╰──────────────────────────────────────────────╯"""
    )
    print(f"模型：{agent.llm.model}    人设：{agent.config.name}")
    print(f"技能：{', '.join(skill.name for skill in agent.skills) or '（无）'}")
    print(f"命令执行：{'开（与控制台版一致：会真的执行 shell 命令！）' if agent.enable_tools else '关（只聊天）'}")
    print(f"本机访问：http://127.0.0.1:{port}")
    if host == "0.0.0.0":
        addresses = lan_ipv4_addresses()
        if addresses:
            print("手机访问（手机与电脑连同一个 Wi-Fi，然后在浏览器输入）：")
            for ip in addresses:
                print(f"   👉 http://{ip}:{port}")
        else:
            print("手机访问：未探测到局域网 IP，可执行 `ipconfig getifaddr en0` 查看本机 IP")
    if agent.enable_tools:
        print("⚠️  注意：同一局域网内的设备都能访问本页并触发命令执行，用完请 Ctrl+C 关闭，或加 --no-tools")
    print("按 Ctrl+C 停止服务\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="手机可用的聊天网页（与控制台版同一个 Agent）")
    parser.add_argument("--host", default=WEB_HOST, help=f"监听地址，默认 {WEB_HOST}")
    parser.add_argument("--port", type=int, default=WEB_PORT, help=f"端口，默认 {WEB_PORT}")
    parser.add_argument("--no-tools", action="store_true", help="禁用命令执行（只聊天）")
    args = parser.parse_args()

    try:
        agent = get_agent(enable_tools=not args.no_tools)  # 启动即校验 Key / 配置
    except LLMError as exc:
        raise SystemExit(str(exc))

    if not (STATIC_DIR / "index.html").is_file():
        raise SystemExit(f"缺少前端文件：{STATIC_DIR / 'index.html'}")

    server = ThreadingHTTPServer((args.host, args.port), ChatHandler)
    server.daemon_threads = True
    print_banner(args.host, args.port, agent)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
