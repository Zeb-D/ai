"""生成结果后处理：清洗 / 术语一致性校验 / 结构化解析。"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, Optional

_LEADING_LABEL = re.compile(r"^\s*(translation|译文|翻译)\s*[:：]\s*", re.IGNORECASE)
_WRAPPING_QUOTES = {'"', "'", "“", "”", "‘", "’", "「", "」", "『", "』"}


def clean(text: str) -> str:
    """去掉模型可能多输出的引号、前缀标签与首尾空白。"""
    result = (text or "").strip()
    if len(result) >= 2 and result[0] == result[-1] and result[0] in _WRAPPING_QUOTES:
        result = result[1:-1].strip()
    result = _LEADING_LABEL.sub("", result)
    return result.strip()


def check_terms(src: str, translation: str, terms: Iterable[Any]) -> list[str]:
    """术语一致性兜底校验：源文命中术语但译文未出现目标译法时告警。"""
    warnings: list[str] = []
    for term in terms:
        src_text = getattr(term, "src", None)
        tgt_text = getattr(term, "tgt", None)
        if not src_text or not tgt_text:
            continue
        if src_text in src and tgt_text not in translation:
            warnings.append(f"术语未命中: {src_text} => {tgt_text}")
    return warnings


def try_parse_json(text: str) -> Optional[dict]:
    """尽力从模型输出中解析 JSON（用于结构化输出场景）。"""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`")
        candidate = candidate.replace("json\n", "", 1).strip()
    try:
        return json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None


def postprocess(text: str, src: str, terms: Iterable[Any], tgt_lang: str) -> dict:
    terms_list = list(terms or [])
    cleaned = clean(text)
    return {
        "translation": cleaned,
        "glossary_hits": [{"src": t.src, "tgt": t.tgt} for t in terms_list],
        "warnings": check_terms(src, cleaned, terms_list),
        "target_lang": tgt_lang,
    }
