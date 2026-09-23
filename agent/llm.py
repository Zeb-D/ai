"""DeepSeek（OpenAI 兼容协议）Chat Completions 客户端 —— 控制台版与网页版**共用这一份实现**。

只用标准库，三种用法：

    llm = DeepSeekLLM()
    llm.chat(messages)                      # 控制台：边收边打印，返回完整文本（main.py 用）
    for piece in llm.stream(messages): ...   # 网页：逐块拿增量，配合 SSE 做打字机效果（web.py 用）
    llm.complete(messages)                   # 一次性拿完整回答

请求本身都是同一个接口：POST {base_url}/chat/completions
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEEPSEEK_TEMPERATURE,
    DEEPSEEK_TIMEOUT,
)


class LLMError(RuntimeError):
    """调用失败（缺 Key / 网络 / 鉴权 / 限流等）。"""


class DeepSeekLLM:
    def __init__(
        self,
        model: str = DEEPSEEK_MODEL,
        api_key: str = DEEPSEEK_API_KEY,
        base_url: str = DEEPSEEK_BASE_URL,
        temperature: float = DEEPSEEK_TEMPERATURE,
        timeout: float = DEEPSEEK_TIMEOUT,
    ) -> None:
        if not api_key:
            raise LLMError(
                "未找到 DEEPSEEK_API_KEY：请复制 agent/.env.example 为 agent/.env 并填入 Key，"
                "或执行 export DEEPSEEK_API_KEY=sk-xxx"
            )
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.timeout = timeout

    def __repr__(self) -> str:
        return f"DeepSeekLLM(model={self.model!r}, base_url={self.base_url!r})"

    # ------------------------------------------------------------ HTTP
    def _post(self, payload: dict[str, Any]):
        request = urllib.request.Request(
            url=f"{self.base_url}/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "text/event-stream" if payload.get("stream") else "application/json",
            },
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=self.timeout)
        except urllib.error.HTTPError as exc:  # 4xx / 5xx，body 里通常写了原因
            detail = exc.read().decode("utf-8", "ignore")
            raise LLMError(f"HTTP {exc.code} {exc.reason}：{detail[:400]}") from exc
        except urllib.error.URLError as exc:
            raise LLMError(f"网络错误：{exc.reason}（可检查 DEEPSEEK_BASE_URL 与代理设置）") from exc

    def _payload(self, messages: list[dict[str, Any]], stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": stream,
        }

    # ------------------------------------------------------------ 流式（SSE 增量）
    def _iter_pieces(self, messages: list[dict[str, Any]]) -> Iterator[dict[str, str]]:
        """逐个产出 {"content": ...} 或 {"reasoning": ...}（deepseek-reasoner 的思维链）。"""
        with self._post(self._payload(messages, stream=True)) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                content = delta.get("content")
                if content:
                    yield {"content": content}
                reasoning = delta.get("reasoning_content")
                if reasoning:
                    yield {"reasoning": reasoning}

    def stream(self, messages: list[dict[str, Any]]) -> Iterator[str]:
        """只产出正文增量（思维链丢弃），供网页版边收边显示。"""
        for piece in self._iter_pieces(messages):
            if "content" in piece:
                yield piece["content"]

    # ------------------------------------------------------------ 一次性
    def complete(self, messages: list[dict[str, Any]]) -> str:
        with self._post(self._payload(messages, stream=False)) as response:
            data = json.loads(response.read().decode("utf-8"))
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"响应缺少 choices：{json.dumps(data, ensure_ascii=False)[:300]}")
        return (choices[0].get("message") or {}).get("content") or ""

    # ------------------------------------------------------------ 控制台友好
    def chat(
        self,
        messages: list[dict[str, Any]],
        stream: bool = True,
        on_text: Callable[[str], None] | None = None,
    ) -> str:
        """边收边输出（默认打印到终端），返回完整回答；reasoner 的思维链以 🧠 前缀展示。"""
        if not stream:
            return self.complete(messages)

        emit = on_text or (lambda text: print(text, end="", flush=True))
        parts: list[str] = []
        reasoning_started = False
        gap_written = False
        for piece in self._iter_pieces(messages):
            if "reasoning" in piece:
                if not reasoning_started:
                    emit("\n🧠 思考：")
                    reasoning_started = True
                emit(piece["reasoning"])
                continue
            if reasoning_started and not gap_written:
                emit("\n\n")
                gap_written = True
            emit(piece["content"])
            parts.append(piece["content"])
        return "".join(parts)
