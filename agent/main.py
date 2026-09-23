"""🖥️ 控制台版入口：while 交互循环 + 「命令: / 完成:」Agent 循环。

Agent 逻辑（协议解析 / 提示词组装 / 循环）都在 [`core.py`](agent/core.py:1)，
网页版 [`web.py`](agent/web.py:1) 用的是**同一套**实现，这里只负责命令行交互与终端渲染。

运行：
    cp agent/.env.example agent/.env     # 填入 DEEPSEEK_API_KEY
    python3 agent/main.py                # 交互模式
    python3 agent/main.py -p "百度一下 deepseek 最新模型版本，给出 3 条来源链接"    # 单次提问

代码骨架：
    while True:                        # 外层：人机交互
        读用户输入 → 追加到 messages
        agent.run(messages, emit)      # 内层：模型决策 → 执行命令 → 结果回填（core.Agent）
"""

from __future__ import annotations

import argparse
import os

from config import AGENT_FILE, WORKSPACE_DIR
from core import Agent, Event, load_agent_config
from llm import DeepSeekLLM, LLMError
from skills import skills_overview

BANNER = r"""
╭──────────────────────────────────────────────╮
│   🐣 入门 Agent Demo（DeepSeek + 命令协议）    │
╰──────────────────────────────────────────────╯
"""

HELP_TEXT = """可用命令：
  /help     显示帮助
  /skills   列出发现的技能文档（agent/skill.md 等）
  /history  查看当前上下文消息条数
  /clear    清空对话历史（保留人设）
  /exit     退出（也可用 quit / q / Ctrl+C）

其它输入会交给模型处理，模型可能输出「命令:」让本机执行 shell 命令。
"""


def make_console_emitter(agent_name: str):
    """把 Agent 事件渲染成终端输出（网页版则渲染成 SSE，见 web.py）。"""

    def emit(event: Event) -> None:
        if event.kind == "start":
            print(f"\n🤖 {agent_name} > ", end="", flush=True)
        elif event.kind == "text":
            print(event.text, end="", flush=True)
        elif event.kind == "end":
            print()
        elif event.kind == "command":
            print(f"   [{event.step}/{event.total_steps}] $ {event.text}")
        elif event.kind == "result":
            print(f"   ↳ {event.text}")
        elif event.kind == "notice":
            print(f"   ⚠️ {event.text}")
        elif event.kind == "error":
            print(f"\n[LLM 调用失败] {event.text}")

    return emit


def handle_command(line: str, agent: Agent, messages: list[dict]) -> bool:
    """处理以 / 开头的本地命令，返回 True 表示退出程序。"""
    command = line.strip().lower()
    if command in {"/exit", "/quit", "exit", "quit", "q"}:
        return True
    if command in {"/help", "/h", "help", "?"}:
        print(HELP_TEXT)
    elif command == "/skills":
        print(skills_overview(agent.skills))
    elif command == "/history":
        print(f"当前上下文消息数：{len(messages)}")
    elif command == "/clear":
        del messages[1:]
        print("已清空对话历史。")
    else:
        print(f"未知命令：{line}，输入 /help 查看可用命令。")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="入门 Agent Demo（DeepSeek + 命令协议）")
    parser.add_argument("-p", "--prompt", help="只问一句就退出，不进入交互模式")
    parser.add_argument("--no-tools", action="store_true", help="禁用命令执行（只聊天、不动本机）")
    args = parser.parse_args()

    os.chdir(WORKSPACE_DIR)  # 让命令与文件路径都相对项目根目录

    if not AGENT_FILE.exists():
        raise SystemExit(f"缺少 Agent 定义文件：{AGENT_FILE}")

    agent_config = load_agent_config(AGENT_FILE)
    try:
        llm = DeepSeekLLM(model=agent_config.model, temperature=agent_config.temperature)
    except LLMError as exc:
        raise SystemExit(str(exc))

    agent = Agent(llm, agent_config, enable_tools=not args.no_tools)
    emit = make_console_emitter(agent_config.name)
    messages: list[dict] = [{"role": "system", "content": agent.system_prompt}]

    if args.prompt:  # 非交互：单次提问
        messages.append({"role": "user", "content": args.prompt})
        agent.run(messages, emit)
        return

    print(BANNER)
    print(f"模型：{llm.model}    工作目录：{WORKSPACE_DIR}    命令执行：{'开' if agent.enable_tools else '关'}")
    print(f"技能：{', '.join(skill.name for skill in agent.skills) or '（无）'}    输入 /help 查看命令，/exit 退出")

    while True:  # 外层循环：持续接收用户输入
        try:
            user_input = input("\n👤 你 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 再见")
            break

        if not user_input:
            continue
        if user_input.startswith("/") or user_input.lower() in {"exit", "quit", "q"}:
            if handle_command(user_input, agent, messages):
                print("👋 再见")
                break
            continue

        messages.append({"role": "user", "content": user_input})
        try:
            agent.run(messages, emit)
        except KeyboardInterrupt:
            print("\n（已中断本轮回答）")


if __name__ == "__main__":
    main()
