"""FastAPI 应用入口：装配配置、引擎、服务、路由与可观测性中间件。

支持两种启动方式：
    1) 作为包运行（推荐）：uvicorn app.main:app   或   python -m app.main
    2) 直接以脚本运行：     python app/main.py（自动补齐包路径）

启动流程（lifespan）：
  load settings -> build engine -> engine.load() -> glossary/cache/metrics/ratelimiter
  -> (可选) micro-batcher -> TranslationService -> 挂载到 app.state
"""

from __future__ import annotations

import sys
from pathlib import Path

# 直接以脚本方式运行（python app/main.py）时，__package__ 为空，相对导入会失败；
# 此处将项目根目录加入 sys.path 并设置包名，使两种启动方式行为一致。
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "app"

import asyncio  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402
import uuid  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request, Response  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from . import __version__  # noqa: E402
from .api import routes_openai, routes_translate  # noqa: E402
from .config import Settings, load_settings  # noqa: E402
from .engine import build_engine  # noqa: E402
from .engine.base import EngineError  # noqa: E402
from .infra.batching import MicroBatcher  # noqa: E402
from .infra.cache import TwoLevelCache  # noqa: E402
from .infra.logging import setup_logging, trace_id_var  # noqa: E402
from .infra.metrics import Metrics  # noqa: E402
from .infra.ratelimit import TokenBucket  # noqa: E402
from .translation.glossary import GlossaryEngine  # noqa: E402
from .translation.prompt import PromptBuilder  # noqa: E402
from .translation.service import TranslationService  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = load_settings()
    setup_logging(settings.serve.log_level)
    logger.info("启动翻译服务", extra={"engine": settings.model.engine})

    # 1) 推理引擎（本地加载权重）
    engine = build_engine(settings.model)
    engine.load()

    # 2) 基础设施
    glossary = GlossaryEngine.from_file(settings.serve.glossary_path)
    cache = TwoLevelCache(
        l1_size=settings.serve.cache.l1_size,
        l2_redis=settings.serve.cache.l2_redis,
        ttl=settings.serve.cache.ttl,
        enabled=settings.serve.cache.enabled,
    )
    metrics = Metrics(enabled=True)
    ratelimiter = TokenBucket(
        qps=settings.serve.ratelimit.qps,
        burst=settings.serve.ratelimit.burst,
        enabled=settings.serve.ratelimit.enabled,
    )

    # 3) 微批处理（可选）
    batcher = None
    if settings.serve.batching.enabled:
        batcher = MicroBatcher(
            engine,
            max_batch_size=settings.serve.batching.max_batch_size,
            max_wait_ms=settings.serve.batching.max_wait_ms,
        )
        await batcher.start()

    # 4) 编排服务
    service = TranslationService(
        engine=engine,
        settings=settings,
        glossary=glossary,
        cache=cache,
        metrics=metrics,
        prompt_builder=PromptBuilder(settings.prompts_dir),
        batcher=batcher,
    )

    app.state.settings = settings
    app.state.engine = engine
    app.state.service = service
    app.state.metrics = metrics
    app.state.ratelimiter = ratelimiter
    app.state.batcher = batcher
    app.state.semaphore = asyncio.Semaphore(settings.serve.max_concurrency)

    logger.info("服务就绪", extra={"engine": engine.name, "glossary_terms": len(glossary)})
    try:
        yield
    finally:
        if batcher is not None:
            await batcher.stop()
        engine.close()
        logger.info("服务已关闭")


app = FastAPI(title="Translation LLM Serving", version=__version__, lifespan=lifespan)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    trace_id = uuid.uuid4().hex[:16]
    token = trace_id_var.set(trace_id)
    start = time.perf_counter()
    status = 500
    try:
        async with app.state.semaphore:
            response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-Id"] = trace_id
        return response
    finally:
        duration = time.perf_counter() - start
        metrics: Metrics = app.state.metrics
        endpoint = request.url.path
        metrics.latency.labels(endpoint=endpoint).observe(duration)
        metrics.requests.labels(endpoint=endpoint, status=str(status)).inc()
        logger.info(
            "http_request",
            extra={
                "endpoint": endpoint,
                "status": status,
                "duration_ms": int(duration * 1000),
                "trace_id": trace_id,
            },
        )
        trace_id_var.reset(token)


@app.exception_handler(EngineError)
async def engine_error_handler(request: Request, exc: EngineError) -> JSONResponse:
    logger.error("引擎错误: %s", exc)
    return JSONResponse(status_code=503, content={"detail": f"推理引擎不可用: {exc}"})


@app.get("/", tags=["meta"])
async def root() -> dict:
    return {"service": "translation-serving", "version": __version__}


@app.get("/healthz", tags=["meta"])
async def healthz() -> dict:
    return {"status": "ok", **app.state.service.health()}


@app.get("/metrics", tags=["meta"])
async def metrics_endpoint() -> Response:
    body, content_type = app.state.metrics.render()
    return Response(content=body, media_type=content_type)


app.include_router(routes_translate.router)
app.include_router(routes_openai.router)


def main() -> None:
    import uvicorn

    settings = load_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.serve.host,
        port=settings.serve.port,
        log_level=settings.serve.log_level.lower(),
    )


if __name__ == "__main__":
    main()
