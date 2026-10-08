"""两级缓存：L1 进程内 LRU + L2 Redis（可选）。

缓存键包含 prompt_version 与 glossary_version，保证 Prompt/术语变更后自动失效。
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from typing import Any, Optional

logger = logging.getLogger(__name__)


def cache_key(
    src: str,
    src_lang: str,
    tgt_lang: str,
    domain: str,
    glossary_version: str = "",
    prompt_version: str = "v1",
) -> str:
    raw = f"{src}|{src_lang}|{tgt_lang}|{domain}|{glossary_version}|{prompt_version}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class TwoLevelCache:
    def __init__(
        self,
        l1_size: int = 4096,
        l2_redis: Optional[str] = None,
        ttl: int = 86400,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self.l1_size = l1_size
        self.ttl = ttl
        self._l1: "OrderedDict[str, Any]" = OrderedDict()
        self._lock = threading.Lock()
        self._redis = None
        if enabled and l2_redis:
            self._init_redis(l2_redis)

    def _init_redis(self, url: str) -> None:
        try:
            import redis

            client = redis.Redis.from_url(url, decode_responses=True)
            client.ping()
            self._redis = client
            logger.info("L2 缓存已启用: %s", url)
        except Exception as exc:  # noqa: BLE001
            logger.warning("L2 Redis 不可用，降级为仅 L1: %s", exc)
            self._redis = None

    # ---- 读写 ----
    def get(self, key: str) -> Optional[Any]:
        if not self.enabled:
            return None
        with self._lock:
            if key in self._l1:
                self._l1.move_to_end(key)
                return self._l1[key]
        if self._redis is not None:
            try:
                raw = self._redis.get(f"ts:{key}")
                if raw:
                    value = json.loads(raw)
                    self._set_l1(key, value)
                    return value
            except Exception:  # noqa: BLE001
                return None
        return None

    def set(self, key: str, value: Any) -> None:
        if not self.enabled:
            return
        self._set_l1(key, value)
        if self._redis is not None:
            try:
                self._redis.setex(f"ts:{key}", self.ttl, json.dumps(value, ensure_ascii=False))
            except Exception:  # noqa: BLE001
                pass

    def _set_l1(self, key: str, value: Any) -> None:
        with self._lock:
            self._l1[key] = value
            self._l1.move_to_end(key)
            while len(self._l1) > self.l1_size:
                self._l1.popitem(last=False)

    # ---- 观测 ----
    def stats(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "l1_size": len(self._l1),
                "l1_capacity": self.l1_size,
                "l2_enabled": self._redis is not None,
            }
