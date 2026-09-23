"""agent demo 配置（控制台版与网页版共用一份）。

- 控制台版入口：`python3 agent/main.py`
- 网页版入口：`python3 agent/web.py`
两者都用 [`llm.py`](agent/llm.py:1) 里同一个 DeepSeek 客户端、同一份 `.env`。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------- 路径
DEMO_DIR = Path(__file__).resolve().parent  # 本 demo 目录：agent/
WORKSPACE_DIR = DEMO_DIR.parent  # 工作区根目录：控制台版命令在这里执行
AGENT_FILE = DEMO_DIR / "agent.md"  # Agent 定义（人设 + 输出协议）
STATIC_DIR = DEMO_DIR / "static"  # 网页版前端文件
ENV_FILE = DEMO_DIR / ".env"  # 控制台版 / 网页版共用的配置


# ---------------------------------------------------------------- .env
def load_dotenv(path: Path) -> bool:
    """极简 .env 解析：`KEY=VALUE`，支持 # 注释，不覆盖已有环境变量。返回是否加载了文件。"""
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


load_dotenv(ENV_FILE)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError:
        return default


# ---------------------------------------------------------------- LLM（控制台版 + 网页版共用）
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEEPSEEK_BASE_URL = (os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1").rstrip("/")
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # 可选 deepseek-reasoner
DEEPSEEK_TEMPERATURE = float(os.getenv("DEEPSEEK_TEMPERATURE", "0.3"))
DEEPSEEK_TIMEOUT = float(os.getenv("DEEPSEEK_TIMEOUT", "120"))

# ---------------------------------------------------------------- 控制台版
MAX_STEPS = _env_int("AGENT_MAX_STEPS", 10)  # 一轮提问内最多执行几条「命令:」
COMMAND_OUTPUT_MAX_CHARS = _env_int("AGENT_COMMAND_OUTPUT_MAX_CHARS", 4000)  # 回填模型的命令输出上限
AGENT_SHELL = os.getenv("AGENT_SHELL", "/bin/bash")  # os.popen 实际调用的 shell

# ---------------------------------------------------------------- 网页版
WEB_HOST = os.getenv("AGENT_WEB_HOST", "0.0.0.0")  # 0.0.0.0 = 手机可在同一局域网访问
WEB_PORT = _env_int("AGENT_WEB_PORT", 8000)
WEB_MAX_SESSIONS = _env_int("AGENT_WEB_MAX_SESSIONS", 50)  # 内存里最多保留多少个会话
WEB_MAX_HISTORY_MESSAGES = _env_int("AGENT_WEB_MAX_HISTORY", 20)  # 每次请求带多少条历史
WEB_MAX_MESSAGE_CHARS = _env_int("AGENT_WEB_MAX_MESSAGE_CHARS", 4000)
# 网页版与控制台版是同一个 Agent（同一份人设 agent.md、同一套技能/命令能力），
# 是否允许网页触发命令执行可用 AGENT_WEB_TOOLS=0 关闭（局域网内任何人都能访问，注意风险）。
WEB_ENABLE_TOOLS = os.getenv("AGENT_WEB_TOOLS", "1").strip().lower() not in {"0", "false", "no", "off"}
# 渠道补充要求：让回答更适合手机小屏与语音朗读
WEB_EXTRA_INSTRUCTIONS = os.getenv(
    "AGENT_WEB_EXTRA_INSTRUCTIONS",
    """## 手机端补充要求

1. 回答面向手机小屏：多用短句、短段落，避免宽表格和复杂排版；
2. 回答可能被语音朗读，请少用 Markdown 符号、表情和缩写；
3. 除非用户明确要求详细展开，否则控制在 200 字以内。""",
)
