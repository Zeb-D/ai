"""OpenAI 兼容接口：/v1/chat/completions。

【合规声明 · 反需求 R1】
本接口由本地引擎直接服务，绝不转发任何外部服务。
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import AsyncIterator, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from ..engine.base import GenRequest
from .schemas import ChatCompletionRequest

router = APIRouter(prefix="/v1", tags=["openai"])


async def _rate_limit(request: Request) -> None:
    limiter = getattr(request.app.state, "ratelimiter", None)
    if limiter is not None and not await limiter.acquire():
        raise HTTPException(
            status_code=429, detail="请求过于频繁，请稍后重试", headers={"Retry-After": "1"}
        )


def _normalize_stop(stop) -> Optional[list[str]]:
    if stop is None:
        return None
    return [stop] if isinstance(stop, str) else list(stop)


def _guided_json(response_format: Optional[dict]) -> Optional[dict]:
    if not response_format:
        return None
    if response_format.get("type") == "json_schema":
        return response_format.get("json_schema", {}).get("schema")
    return None


def _build_gen_request(payload: ChatCompletionRequest, request: Request) -> GenRequest:
    settings = request.app.state.settings
    max_tokens = payload.max_tokens or settings.model.default_max_tokens
    return GenRequest(
        messages=[m.model_dump() for m in payload.messages],
        max_tokens=max_tokens,
        temperature=payload.temperature,
        top_p=payload.top_p,
        stop=_normalize_stop(payload.stop),
        guided_json=_guided_json(payload.response_format),
    )


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request):
    await _rate_limit(request)
    engine = request.app.state.engine
    model_name = request.app.state.settings.model.model_path
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    gen_req = _build_gen_request(payload, request)

    # ---- 流式 ----
    if payload.stream:

        async def event_generator() -> AsyncIterator[str]:
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
            )
            try:
                async for delta in engine.stream(gen_req):
                    yield _sse(
                        {
                            "id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": model_name,
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"content": delta},
                                    "finish_reason": None,
                                }
                            ],
                        }
                    )
            except Exception as exc:  # noqa: BLE001
                yield _sse({"error": str(exc)})
            yield _sse(
                {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                }
            )
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ---- 非流式 ----
    result = await asyncio.to_thread(engine.generate, gen_req)
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }
