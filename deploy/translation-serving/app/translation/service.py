"""翻译编排服务：把业务翻译意图翻译成模型生成请求，并处理术语/缓存/后处理。

- 计算密集的同步引擎调用通过 asyncio.to_thread 卸载，避免阻塞事件循环；
- 支持单条 / 批量 / 流式三种模式；
- 开启 batching 时，单条请求经 MicroBatcher 聚合为批。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import AsyncIterator, Optional, Sequence

from ..config import Settings
from ..engine.base import GenRequest, GenResult, InferenceEngine
from ..infra.batching import MicroBatcher
from ..infra.cache import TwoLevelCache, cache_key
from ..infra.metrics import Metrics
from .glossary import GlossaryEngine
from .postprocess import postprocess
from .prompt import PromptBuilder

logger = logging.getLogger(__name__)


class TranslationService:
    def __init__(
        self,
        engine: InferenceEngine,
        settings: Settings,
        glossary: GlossaryEngine,
        cache: TwoLevelCache,
        metrics: Metrics,
        prompt_builder: Optional[PromptBuilder] = None,
        batcher: Optional[MicroBatcher] = None,
    ):
        self.engine = engine
        self.settings = settings
        self.glossary = glossary
        self.cache = cache
        self.metrics = metrics
        self.prompt_builder = prompt_builder or PromptBuilder(settings.prompts_dir)
        self.batcher = batcher

    # ------------------------------------------------------------------
    # 内部：构建请求 + 生成
    # ------------------------------------------------------------------
    def _key(self, src, src_lang, tgt_lang, domain, use_glossary) -> str:
        glossary_version = self.glossary.version if use_glossary else ""
        return cache_key(
            src, src_lang, tgt_lang, domain, glossary_version, self.prompt_builder.version
        )

    async def _generate(self, req: GenRequest) -> GenResult:
        if self.batcher is not None and self.settings.serve.batching.enabled:
            return await self.batcher.submit(req)
        return await asyncio.to_thread(self.engine.generate, req)

    # ------------------------------------------------------------------
    # 单条翻译
    # ------------------------------------------------------------------
    async def translate(
        self,
        src: str,
        src_lang: str,
        tgt_lang: str,
        domain: str = "general",
        use_glossary: bool = True,
    ) -> dict:
        key = self._key(src, src_lang, tgt_lang, domain, use_glossary)
        cached = self.cache.get(key)
        if cached is not None:
            self.metrics.cache_hits.inc()
            return {**cached, "cached": True, "prompt_tokens": 0, "completion_tokens": 0}

        self.metrics.cache_misses.inc()
        terms = self.glossary.match(src, domain) if use_glossary else []
        messages = self.prompt_builder.build_messages(src, src_lang, tgt_lang, domain, terms)
        req = GenRequest(
            messages=messages,
            temperature=self.settings.model.default_temperature,
            max_tokens=self.settings.model.default_max_tokens,
        )
        result = await self._generate(req)

        out = postprocess(result.text, src, terms, tgt_lang)
        self.metrics.tokens.inc(result.completion_tokens)
        self.metrics.prompt_tokens.inc(result.prompt_tokens)

        payload = {
            "translation": out["translation"],
            "glossary_hits": out["glossary_hits"],
            "warnings": out["warnings"],
            "target_lang": out["target_lang"],
        }
        self.cache.set(key, payload)
        return {
            **payload,
            "cached": False,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
        }

    # ------------------------------------------------------------------
    # 批量翻译
    # ------------------------------------------------------------------
    async def translate_many(
        self,
        texts: Sequence[str],
        src_lang: str,
        tgt_lang: str,
        domain: str = "general",
        use_glossary: bool = True,
    ) -> list[dict]:
        tasks = [
            self.translate(text, src_lang, tgt_lang, domain, use_glossary) for text in texts
        ]
        return list(await asyncio.gather(*tasks))

    # ------------------------------------------------------------------
    # 流式翻译
    # ------------------------------------------------------------------
    async def translate_stream(
        self,
        src: str,
        src_lang: str,
        tgt_lang: str,
        domain: str = "general",
        use_glossary: bool = True,
    ) -> AsyncIterator[str]:
        terms = self.glossary.match(src, domain) if use_glossary else []
        messages = self.prompt_builder.build_messages(src, src_lang, tgt_lang, domain, terms)
        req = GenRequest(
            messages=messages,
            temperature=self.settings.model.default_temperature,
            max_tokens=self.settings.model.default_max_tokens,
        )
        collected: list[str] = []
        async for delta in self.engine.stream(req):
            collected.append(delta)
            yield delta
        # 完成后写入缓存（缓存清洗后的完整译文）
        full = "".join(collected)
        out = postprocess(full, src, terms, tgt_lang)
        self.cache.set(
            self._key(src, src_lang, tgt_lang, domain, use_glossary),
            {
                "translation": out["translation"],
                "glossary_hits": out["glossary_hits"],
                "warnings": out["warnings"],
                "target_lang": out["target_lang"],
            },
        )

    # ------------------------------------------------------------------
    # 健康检查
    # ------------------------------------------------------------------
    def health(self) -> dict:
        status = {
            "engine": self.engine.health(),
            "cache": self.cache.stats(),
            "glossary_terms": len(self.glossary),
            "glossary_version": self.glossary.version,
            "prompt_version": self.prompt_builder.version,
        }
        if self.batcher is not None:
            status["batcher_depth"] = self.batcher.depth
        return status
