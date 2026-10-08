"""异步令牌桶限流。

超限时由路由层返回 429（含 Retry-After 建议）。
"""

from __future__ import annotations

import asyncio
import time


class TokenBucket:
    def __init__(self, qps: float, burst: int, enabled: bool = True):
        self.qps = float(qps)
        self.burst = float(burst)
        self.enabled = enabled
        self._tokens = float(burst)
        self._last = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self, amount: float = 1.0) -> bool:
        if not self.enabled:
            return True
        async with self._lock:
            now = time.monotonic()
            elapsed = now - self._last
            self._last = now
            self._tokens = min(self.burst, self._tokens + elapsed * self.qps)
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

    @property
    def available(self) -> float:
        return self._tokens
