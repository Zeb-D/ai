"""命令执行层：Agent 唯一的「手脚」（控制台版与网页版共用）。

模型只会吐出文本「命令:XXX」，真正干活发生在这里 —— 用 os.popen 执行 shell 命令，
把 stdout + stderr 作为「观察结果」回填给模型，模型据此决定下一步。

为什么用 os.popen？
- 一行就能跑命令拿输出，最适合入门 demo 看清「工具层」的本质；
- 代价：拿不到退出码、无法设置超时、没有沙箱隔离。
  生产环境应换成 subprocess.run(..., timeout=..., shell=True) + 容器/白名单隔离。
"""

from __future__ import annotations

import os
import re
import shlex

from config import AGENT_SHELL, COMMAND_OUTPUT_MAX_CHARS, WORKSPACE_DIR

# 只是最粗的一层护栏：命中即拒绝执行。
# 注意它挡不住有心人（真正的安全边界必须是容器/低权限账户/白名单）。
BLOCKED_PATTERNS = [
    r"\bsudo\b",
    r"\brm\s+(-[a-zA-Z]+\s+)*(-rf|-fr)\s+/(\s|$)",
    r"\bmkfs(\.\w+)?\b",
    r"\bdd\s+.*of=/dev/",
    r"\b(shutdown|reboot|halt)\b",
    r":\(\)\s*\{",  # fork bomb
    r">\s*/dev/(sd|disk|rdisk)",
]


def _guard(command: str) -> str | None:
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return f"[已拦截] 命令命中危险模式 `{pattern}`，不予执行：{command}"
    return None


def _truncate(text: str, limit: int = COMMAND_OUTPUT_MAX_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...（输出过长已截断，完整长度 {len(text)} 字符）"


def run_command(command: str, max_chars: int = COMMAND_OUTPUT_MAX_CHARS) -> str:
    """执行一条 shell 命令，返回可读的输出文本（stdout + stderr）。

    - 命令固定在工作区根目录下执行，所以提示里可以直接写 `cat agent/skill.md`；
    - `2>&1` 让 stderr 一起回填，模型才能看到报错并自行修正；
    - 任何异常都转成字符串返回，保证 Agent 循环不中断。
    """
    command = (command or "").strip()
    if not command:
        return "[错误] 收到空命令，请重新输出 命令:XXX"

    blocked = _guard(command)
    if blocked:
        return blocked

    # os.popen 只捕获 stdout，这里用 `2>&1` 合并 stderr；cwd 通过 cd 显式指定。
    full_command = f"cd {shlex.quote(str(WORKSPACE_DIR))} && {AGENT_SHELL} -c {shlex.quote(command)} 2>&1"

    try:
        with os.popen(full_command) as pipe:  # 相当于 file = os.popen(...)
            output = pipe.read()
    except OSError as exc:
        return f"[错误] 无法启动命令：{exc}"
    except Exception as exc:  # noqa: BLE001 - 兜底，保证循环不崩
        return f"[错误] {type(exc).__name__}: {exc}"

    if not output.strip():
        return "(命令执行完成，但没有输出)"
    return _truncate(output.strip("\n"), max_chars)
