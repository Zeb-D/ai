"""微批处理聚合器。

把高并发到达的单条请求在很短的等待窗口内聚合成批，一次性交给引擎的
generate_batch（离线/文档翻译场景可显著提升 GPU 利用率）。
在线高并发场景通常无需开启——引擎自带的 Continuous Batching 已足够。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

from ..engine.base import GenRequest, GenResult, InferenceEngine

logger = logging.getLogger(__name__)


@dataclass
class _Pending:
    req: GenRequest
    future: "asyncio.Future[GenResult]"


class MicroBatcher:
    def __init__(
        self,
        engine: InferenceEngine,
        max_batch_size: int = 16,
        max_wait_ms: int = 10,
    ):
        self.engine = engine
        self.max_batch_size = max_batch_size
        self.max_wait_s = max_wait_ms / 1000.0
        self._queue: Optional[asyncio.Queue] = None
        self._worker: Optional[asyncio.Task] = None
        self.closed = False

    async def start(self) -> None:
        if self._worker is None:
            self._queue = asyncio.Queue()
            self._worker = asyncio.create_task(self._run(), name="micro-batcher")

    async def stop(self) -> None:
        self.closed = True
        if self._worker is not None:
            self._worker.cancel()
            try:
                await self._worker
            except asyncio.CancelledError:
                pass
            self._worker = None

    async def submit(self, req: GenRequest) -> GenResult:
        if self._queue is None or self._worker is None:
            await self.start()
        assert self._queue is not None
        loop = asyncio.get_running_loop()
        fut: "asyncio.Future[GenResult]" = loop.create_future()
        await self._queue.put(_Pending(req, fut))
        return await fut

    @property
    def depth(self) -> int:
        return self._queue.qsize() if self._queue is not None else 0

    async def _run(self) -> None:
        assert self._queue is not None
        while True:
            first = await self._queue.get()
            batch = [first]
            deadline = time.monotonic() + self.max_wait_s
            while len(batch) < self.max_batch_size:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break
            await self._dispatch(batch)

    async def _dispatch(self, batch: list[_Pending]) -> None:
        reqs = [p.req for p in batch]
        try:
            results = await asyncio.to_thread(self.engine.generate_batch, reqs)
            for pending, result in zip(batch, results):
                if not pending.future.done():
                    pending.future.set_result(result)
        except Exception as exc:  # noqa: BLE001
            logger.exception("批处理分发失败: %s", exc)
            for pending in batch:
                if not pending.future.done():
                    pending.future.set_exception(exc)
