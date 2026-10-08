"""Prometheus 指标。

未安装 prometheus-client 时自动降级为 no-op，保证服务仍可运行。
"""

from __future__ import annotations

try:  # pragma: no cover - 取决于环境
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
    )

    _HAS_PROM = True
except ImportError:  # pragma: no cover
    _HAS_PROM = False
    CONTENT_TYPE_LATEST = "text/plain; version=0.0.4"


class _Noop:
    """no-op 指标替身。"""

    def inc(self, *args, **kwargs) -> None:
        return None

    def observe(self, *args, **kwargs) -> None:
        return None

    def set(self, *args, **kwargs) -> None:
        return None

    def labels(self, *args, **kwargs) -> "_Noop":
        return self


class Metrics:
    """统一指标门面。"""

    def __init__(self, enabled: bool = True):
        use = enabled and _HAS_PROM
        self.enabled = use
        if use:
            self.requests = Counter(
                "translation_requests_total", "请求总数", ["endpoint", "status"]
            )
            self.latency = Histogram(
                "translation_latency_seconds", "请求耗时(秒)", ["endpoint"]
            )
            self.tokens = Counter("tokens_completion_total", "生成 token 总数")
            self.prompt_tokens = Counter("tokens_prompt_total", "Prompt token 总数")
            self.cache_hits = Counter("translation_cache_hits_total", "缓存命中数")
            self.cache_misses = Counter("translation_cache_misses_total", "缓存未命中数")
            self.inflight = Gauge("translation_inflight_requests", "处理中的请求数")
            self.queue_depth = Gauge("translation_queue_depth", "批处理队列深度")
        else:
            self.requests = _Noop()
            self.latency = _Noop()
            self.tokens = _Noop()
            self.prompt_tokens = _Noop()
            self.cache_hits = _Noop()
            self.cache_misses = _Noop()
            self.inflight = _Noop()
            self.queue_depth = _Noop()

    def render(self) -> tuple[bytes, str]:
        if self.enabled:
            return generate_latest(), CONTENT_TYPE_LATEST
        return b"", "text/plain; charset=utf-8"
