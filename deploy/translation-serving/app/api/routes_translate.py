"""业务翻译接口：/v1/translate 与 /v1/translate/stream。"""

from __future__ import annotations

import json
import time
import uuid
from typing import AsyncIterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..infra.logging import trace_id_var
from .schemas import (
    GlossaryHit,
    TranslateRequest,
    TranslateResponse,
    TranslationItem,
    TranslationMeta,
)

router = APIRouter(prefix="/v1", tags=["translate"])


async def _rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "ratelimiter", None)
    if limiter is not None and not await limiter.acquire():
        raise HTTPException(
            status_code=429, detail="请求过于频繁，请稍后重试", headers={"Retry-After": "1"}
        )


@router.post("/translate", response_model=TranslateResponse)
async def translate(payload: TranslateRequest, request: Request) -> TranslateResponse:
    await _rate_limit(request)
    service = request.app.state.service
    metrics = request.app.state.metrics
    settings = request.app.state.settings

    request_id = trace_id_var.get()
    if request_id in ("", "-"):
        request_id = uuid.uuid4().hex[:16]

    texts = [payload.text] if isinstance(payload.text, str) else list(payload.text)
    if not texts:
        raise HTTPException(status_code=400, detail="text 不能为空")

    metrics.inflight.inc()
    start = time.perf_counter()
    try:
        results = await service.translate_many(
            texts,
            payload.source_lang,
            payload.target_lang,
            payload.domain,
            payload.use_glossary,
        )
    finally:
        metrics.inflight.dec()
    duration_ms = int((time.perf_counter() - start) * 1000)

    items = [
        TranslationItem(
            translation=r["translation"],
            cached=r["cached"],
            glossary_hits=[GlossaryHit(**hit) for hit in r["glossary_hits"]],
            warnings=r["warnings"],
            prompt_tokens=r["prompt_tokens"],
            completion_tokens=r["completion_tokens"],
        )
        for r in results
    ]
    meta = TranslationMeta(
        model=settings.model.model_path,
        engine=request.app.state.engine.name,
        request_id=request_id,
        duration_ms=duration_ms,
    )
    return TranslateResponse(
        translations=[r["translation"] for r in results], items=items, meta=meta
    )


@router.post("/translate/stream")
async def translate_stream(payload: TranslateRequest, request: Request) -> StreamingResponse:
    await _rate_limit(request)
    if not isinstance(payload.text, str):
        raise HTTPException(status_code=400, detail="流式接口仅支持单条文本")
    service = request.app.state.service

    async def event_generator() -> AsyncIterator[str]:
        try:
            async for delta in service.translate_stream(
                payload.text,
                payload.source_lang,
                payload.target_lang,
                payload.domain,
                payload.use_glossary,
            ):
                yield f"data: {json.dumps({'delta': delta}, ensure_ascii=False)}\n\n"
        except Exception as exc:  # noqa: BLE001 - 流内错误以事件返回，避免连接中断
            yield f"data: {json.dumps({'error': str(exc)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
