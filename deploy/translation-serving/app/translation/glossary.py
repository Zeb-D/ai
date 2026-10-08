"""术语约束引擎。

策略（见架构文档 §3.7）：Prompt 注入为主，后处理一致性校验为兜底。
术语表支持 yaml / json，结构示例见 configs/glossary.example.yaml。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import yaml


@dataclass(frozen=True)
class Term:
    src: str
    tgt: str
    domain: str = "general"
    case_sensitive: bool = False


class GlossaryEngine:
    def __init__(self, terms: Optional[Iterable[Term]] = None):
        self._terms: list[Term] = list(terms or [])
        self._version = self._compute_version()

    # ---- 构建 ----
    @classmethod
    def from_file(cls, path: str | Path | None) -> "GlossaryEngine":
        if not path:
            return cls([])
        p = Path(path)
        if not p.exists():
            return cls([])
        text = p.read_text(encoding="utf-8")
        data = yaml.safe_load(text) if p.suffix in (".yaml", ".yml") else json.loads(text)
        raw_terms = data.get("terms", []) if isinstance(data, dict) else (data or [])
        terms = [
            Term(
                src=str(t["src"]),
                tgt=str(t["tgt"]),
                domain=str(t.get("domain", "general")),
                case_sensitive=bool(t.get("case_sensitive", False)),
            )
            for t in raw_terms
        ]
        return cls(terms)

    def _compute_version(self) -> str:
        payload = "|".join(
            sorted(f"{t.src}=>{t.tgt}@{t.domain}@{int(t.case_sensitive)}" for t in self._terms)
        )
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]

    # ---- 访问 ----
    @property
    def version(self) -> str:
        return self._version

    @property
    def terms(self) -> list[Term]:
        return list(self._terms)

    def __len__(self) -> int:
        return len(self._terms)

    # ---- 匹配 ----
    def match(self, text: str, domain: str = "general") -> list[Term]:
        """返回在 text 中命中、且适用于 domain 的术语（长词优先，避免子串误伤）。"""
        hits: list[Term] = []
        for term in self._terms:
            if term.domain not in (domain, "general"):
                continue
            if term.case_sensitive:
                found = term.src in text
            else:
                found = term.src.lower() in text.lower()
            if found:
                hits.append(term)
        hits.sort(key=lambda t: len(t.src), reverse=True)
        return hits
