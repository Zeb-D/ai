"""Agent 核心逻辑（控制台版 `main.py` 与网页版 `web.py` **共用同一套聊天/技能/工具实现**）。

三部分：
1. 协议解析 [`parse_reply()`](agent/core.py:57)：从模型回复里取「命令:XXX」或「完成:XXX」；
2. 提示词组装 [`build_system_prompt()`](agent/core.py:80)：agent.md 正文 + 命令能力 + 技能清单 + 运行环境 + 协议强调；
3. Agent 循环 [`Agent.run()`](agent/core.py:143)：模型决策 → `os.popen` 执行命令 → 结果回填 → 直到「完成:」或预算用尽。

循环过程中通过 `on_event(Event)` 回调抛出事件，两个入口各自渲染：
- 控制台版：`text` 直接打印、`command` 打印 `[步数/上限] $ cmd`；
- 网页版：转成 SSE 的 `data: {"delta": ...}`，手机端边收边显示。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from config import AGENT_FILE, DEMO_DIR, DEEPSEEK_MODEL, MAX_STEPS
from llm import DeepSeekLLM, LLMError
from skills import Skill, discover_skills, load_markdown, skills_overview
from tools import run_command

# 匹配「命令:」/「完成:」标记，容忍模型加上 # > * ` 或【】等装饰
REPLY_RE = re.compile(r"(?m)^[ \t>*#`\-]*[【\[]?\s*(命令|完成)\s*[】\]]?\s*[::]\s*")
PREVIEW_CHARS = 300  # 回显给用户看的命令结果预览长度（完整结果仍会回填给模型）


# ------------------------------------------------------------------ 事件
@dataclass
class Event:
    """Agent 循环抛出的事件：kind ∈ start / text / end / command / result / notice / error。"""

    kind: str
    text: str = ""
    step: int = 0
    total_steps: int = 0
    extra: dict[str, Any] = field(default_factory=dict)


Emit = Callable[[Event], None]


# ------------------------------------------------------------------ 协议解析
def _strip_code_fence(text: str) -> str:
    """模型偶尔会把内容包在 ``` 里，这里剥掉一层。"""
    text = text.strip()
    if not text.startswith("```"):
        return text
    newline = text.find("\n")
    if newline != -1:
        text = text[newline + 1:]
    if text.rstrip().endswith("```"):
        text = text.rstrip()[:-3]
    return text.strip()


def parse_reply(reply: str) -> tuple[str, str] | None:
    """把模型回复解析成 ("命令"|"完成", 载荷)；不符合协议返回 None。"""
    cleaned = _strip_code_fence(reply)
    match = REPLY_RE.search(cleaned)
    if not match:
        return None
    # 取从标记到结尾的全部内容：命令可能是多行的（例如 python3 -c / 管道 / heredoc）
    payload = _strip_code_fence(cleaned[match.end():])
    return match.group(1), payload


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit] + "..."


# ------------------------------------------------------------------ Agent 定义
@dataclass
class AgentConfig:
    """从 agent.md 读出来的东西：人设正文 + front-matter 参数。"""

    name: str
    model: str
    temperature: float
    body: str
    meta: dict[str, Any]


def load_agent_config(path: Path = AGENT_FILE) -> AgentConfig:
    meta, body = load_markdown(path)
    return AgentConfig(
        name=str(meta.get("name") or "assistant"),
        model=str(meta.get("model") or DEEPSEEK_MODEL),
        temperature=float(meta.get("temperature", 0.3)),
        body=body,
        meta=meta,
    )


# ------------------------------------------------------------------ 提示词
def build_system_prompt(
    config: AgentConfig,
    skills: list[Skill],
    *,
    tools_enabled: bool = True,
    extra_instructions: str = "",
) -> str:
    """system prompt = agent.md 正文 + 命令能力 + 技能清单 + 运行环境 + 协议强调（+ 渠道补充）。"""
    sections = [config.body]

    if tools_enabled:
        sections.append(
            "## 你可以执行的命令\n\n"
            "你输出的「命令:」会被系统用 `os.popen` 在项目根目录下执行（可用 `cat / grep / awk / sed / wc / "
            "sort / curl / python3` 等），执行结果会作为「命令执行结果」回给你，你再据此继续推理。"
        )
    else:
        sections.append("## 命令执行\n\n当前环境**不允许执行命令**，请只根据已给信息作答，并始终用「完成:」收尾。")

    sections.append(
        "## 技能清单\n\n"
        + skills_overview(skills)
        + "\n\n需要某个技能时，先 `cat 技能文档路径` 把全文读进来，再严格按文档中的命令步骤执行；"
        "如果路径不存在，先用 `ls` 确认实际路径。"
    )

    sections.append(
        "## 运行环境\n\n"
        f"- 工作目录：`{DEMO_DIR.parent}`（你的每条命令都在这里执行，路径按相对项目根目录书写）\n"
        f"- 每轮提问最多执行 {MAX_STEPS} 条命令\n"
        f"- Agent 人设文件：`{AGENT_FILE}`；技能文档目录：`{DEMO_DIR}`"
    )

    if extra_instructions.strip():
        sections.append(extra_instructions.strip())

    sections.append(
        "## 输出格式（必须严格遵守）\n\n"
        "你的目标是完成用户的任务，你必须选择下面的其中一种格式进行回复:\n\n"
        "1.如果你认为需要执行命令，则输出'命令:XXX'，XXX 为命令本身，不要用任何的格式，不要解释\n\n"
        "2，如果你认为不需要执行命令，则输出'完成:XXX'，XXX 为你的总结信息"
    )
    return "\n\n".join(sections)


# ------------------------------------------------------------------ Agent 循环
class Agent:
    """带技能/命令能力的 Agent：消息数组由调用方持有（控制台单会话、网页按 session 存）。"""

    def __init__(
        self,
        llm: DeepSeekLLM,
        config: AgentConfig | None = None,
        *,
        enable_tools: bool = True,
        max_steps: int = MAX_STEPS,
        extra_instructions: str = "",
    ) -> None:
        self.llm = llm
        self.config = config or load_agent_config()
        self.skills = discover_skills(DEMO_DIR)
        self.enable_tools = enable_tools
        self.max_steps = max_steps
        self.system_prompt = build_system_prompt(
            self.config, self.skills, tools_enabled=enable_tools, extra_instructions=extra_instructions
        )

    # ------------------------------------------------------------ 主循环
    def run(self, messages: list[dict[str, Any]], emit: Emit | None = None) -> str:
        """跑一轮对话（会就地修改 messages）。返回「完成:」后面的总结文本，未收尾则返回空串。"""
        notify: Emit = emit or (lambda event: None)

        for step in range(1, self.max_steps + 1):
            reply = self._ask(messages, notify, step)
            if reply is None:
                return ""
            if not reply.strip():
                notify(Event("notice", "模型返回了空内容，本轮结束"))
                return ""

            messages.append({"role": "assistant", "content": reply})
            parsed = parse_reply(reply)
            if parsed is None:
                notify(Event("notice", "模型未按协议输出（既没有「命令:」也没有「完成:」），按完成处理"))
                return reply

            kind, payload = parsed
            if kind == "完成":
                return payload
            if not payload:
                notify(Event("notice", "解析到的命令为空，本轮结束"))
                return ""

            notify(Event("command", payload, step=step, total_steps=self.max_steps))
            if not self.enable_tools:
                result = "[已禁用] 当前环境不允许执行命令"
            else:
                result = run_command(payload)
            notify(Event("result", preview(result)))
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"命令:{payload}\n\n命令执行结果:\n{result}\n\n"
                        f"(提示：本轮命令预算已用 {step}/{self.max_steps} 条)"
                    ),
                }
            )

        return self._force_finish(messages, notify)

    # ------------------------------------------------------------ 内部
    def _ask(self, messages: list[dict[str, Any]], notify: Emit, step: int) -> str | None:
        """调用一次模型并把文本流通过事件推出去；失败返回 None。"""
        notify(Event("start", step=step, total_steps=self.max_steps))
        try:
            reply = self.llm.chat(messages, stream=True, on_text=lambda piece: notify(Event("text", piece)))
        except LLMError as exc:
            notify(Event("end", step=step, total_steps=self.max_steps))
            notify(Event("error", str(exc)))
            return None
        notify(Event("end", step=step, total_steps=self.max_steps))
        return reply.strip()

    def _force_finish(self, messages: list[dict[str, Any]], notify: Emit) -> str:
        """预算用尽：要求模型立刻用已有信息收尾，避免「转圈没有结论」。"""
        notify(Event("notice", f"已达到单轮命令上限（{self.max_steps} 条），要求模型直接用已有信息收尾"))
        messages.append(
            {
                "role": "user",
                "content": (
                    "命令预算已用尽，不要再执行命令，请立刻输出'完成:XXX'，"
                    "用已经拿到的信息总结（并说明哪些结论不确定）。"
                ),
            }
        )
        reply = self._ask(messages, notify, self.max_steps)
        if not reply:
            return ""
        messages.append({"role": "assistant", "content": reply})
        parsed = parse_reply(reply)
        if parsed and parsed[0] == "完成":
            return parsed[1]
        notify(Event("notice", "模型仍未按协议输出「完成:」"))
        return ""
