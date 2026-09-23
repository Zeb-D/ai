"""技能（skill）发现：扫描 markdown 的 front-matter，把「技能清单」注入系统提示。

约定：任何 markdown 只要带上这段头信息，就是一个技能：

    ---
    type: skill
    name: text_stats
    description: 一句话说明这个技能能干什么
    ---
    # 正文：什么时候用、按顺序执行哪些命令、输出要求

关键设计（渐进式披露 / progressive disclosure）：
- **摘要**（name + description + 文件路径）常驻 system prompt，让模型知道「有这个技能」；
- **正文**不进 prompt，模型需要时自己 `cat agent/skill.md` 读进来 —— 上下文省下来了，
  而且「要不要用这个技能」的决定权交给模型，这正是 Agent 和固定 workflow 的区别。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import WORKSPACE_DIR

FRONT_MATTER_RE = re.compile(r"\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*\r?\n?", re.S)


def _coerce(value: str) -> Any:
    """把 front-matter 的字符串转成 bool / int / float / list / 字符串。"""
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if value.startswith("[") and value.endswith("]"):
        return [_coerce(item) for item in value[1:-1].split(",") if item.strip()]
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def parse_markdown(text: str) -> tuple[dict[str, Any], str]:
    """拆出 front-matter（只支持 `key: value` 与 `[a, b]` 单行写法）和正文。"""
    match = FRONT_MATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    meta: dict[str, Any] = {}
    for raw_line in match.group(1).splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, _, value = line.partition(":")
        meta[key.strip()] = _coerce(value)
    return meta, text[match.end():].strip()


def load_markdown(path: Path) -> tuple[dict[str, Any], str]:
    return parse_markdown(path.read_text(encoding="utf-8"))


@dataclass
class Skill:
    """一个技能 = front-matter 元信息 + markdown 正文。"""

    name: str
    description: str
    path: Path
    meta: dict[str, Any] = field(default_factory=dict)
    body: str = ""

    @property
    def relative_path(self) -> str:
        """展示/执行用的相对路径，例如 agent/skill.md（命令就在工作区根目录执行）。"""
        try:
            return str(self.path.relative_to(WORKSPACE_DIR))
        except ValueError:
            return str(self.path)


def discover_skills(directory: Path) -> list[Skill]:
    """扫描目录（含 skills/ 子目录）下所有 `type: skill` 的 markdown。"""
    candidates = [*directory.glob("*.md"), *directory.glob("skills/*.md")]
    skills: list[Skill] = []
    for path in sorted(candidates):
        try:
            meta, body = load_markdown(path)
        except OSError:
            continue
        if meta.get("type") != "skill":
            continue  # agent.md 之类不是技能
        skills.append(
            Skill(
                name=str(meta.get("name") or path.stem),
                description=str(meta.get("description", "（未填写 description）")),
                path=path,
                meta=meta,
                body=body,
            )
        )
    return skills


def skills_overview(skills: list[Skill]) -> str:
    """把技能清单渲染成给模型看的一段说明。"""
    if not skills:
        return "（当前没有可用技能文档）"
    lines = [
        "| 技能名 | 用途 | 文档路径（用 `cat 路径` 阅读全文） |",
        "| --- | --- | --- |",
    ]
    for skill in skills:
        lines.append(f"| {skill.name} | {skill.description} | `{skill.relative_path}` |")
    return "\n".join(lines)
