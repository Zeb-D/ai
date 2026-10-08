"""Prompt 引擎。

设计要点（影响前缀缓存命中率）：
- System Prompt 的"固定部分"（角色 + 规则）逐字节稳定；
- 术语约束作为可变部分追加在末尾；
- 模板按"翻译方向"（如 zh2en）组织，加载自 configs/prompts/*.jinja，
  缺失时回退到内置默认模板，保证服务永远可用。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional

from jinja2 import Environment, FileSystemLoader, Template

# Prompt 版本：模板语义变更时必须递增（用于缓存键失效）
PROMPT_VERSION = "v1"

LANG_NAMES = {
    "zh": "Chinese",
    "en": "English",
    "ja": "Japanese",
    "ko": "Korean",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ru": "Russian",
    "pt": "Portuguese",
    "it": "Italian",
}

# 内置兜底模板（当 configs/prompts 缺失时使用）
DEFAULT_TEMPLATE = """You are a professional translator. Translate the user's text from {{ src_name }} to {{ tgt_name }}.

Rules:
1. Output ONLY the translation, with no explanation, no notes, and no surrounding quotes.
2. Preserve numbers, units, code, URLs, email addresses, and proper nouns.
3. Keep the translation faithful, natural and concise.
4. Domain: {{ domain }}.
{%- if terms %}
5. You MUST use the following term mappings exactly (source => target):
{%- for t in terms %}
   - "{{ t.src }}" => "{{ t.tgt }}"
{%- endfor %}
{%- endif %}"""


class PromptBuilder:
    def __init__(self, prompts_dir: str | Path | None = None):
        self.prompts_dir = Path(prompts_dir) if prompts_dir else None
        self._env: Optional[Environment] = None
        if self.prompts_dir and self.prompts_dir.exists():
            self._env = Environment(
                loader=FileSystemLoader(str(self.prompts_dir)),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        self._template_cache: dict[str, Template] = {}

    @property
    def version(self) -> str:
        return PROMPT_VERSION

    def _get_template(self, direction: str) -> Template:
        if direction in self._template_cache:
            return self._template_cache[direction]
        template: Optional[Template] = None
        if self._env is not None:
            try:
                template = self._env.get_template(f"{direction}.jinja")
            except Exception:  # noqa: BLE001 - 缺失则回退
                template = None
        if template is None:
            template = Template(DEFAULT_TEMPLATE)
        self._template_cache[direction] = template
        return template

    @staticmethod
    def _lang_name(code: str) -> str:
        return LANG_NAMES.get(code.lower(), code)

    def build_system(
        self,
        src_lang: str,
        tgt_lang: str,
        domain: str = "general",
        terms: Iterable[Any] | None = None,
    ) -> str:
        direction = f"{src_lang.lower()}2{tgt_lang.lower()}"
        template = self._get_template(direction)
        rendered = template.render(
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            src_name=self._lang_name(src_lang),
            tgt_name=self._lang_name(tgt_lang),
            domain=domain or "general",
            terms=list(terms or []),
        )
        return rendered.strip()

    def build_messages(
        self,
        src_text: str,
        src_lang: str,
        tgt_lang: str,
        domain: str = "general",
        terms: Iterable[Any] | None = None,
    ) -> list[dict]:
        system = self.build_system(src_lang, tgt_lang, domain, terms)
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": src_text},
        ]
