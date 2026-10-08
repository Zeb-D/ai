"""结构化 JSON 日志。

便于 Loki/ELK 采集；并支持通过 contextvar 传递 trace_id 贯穿单次请求。
"""

from __future__ import annotations

import json
import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone

trace_id_var: ContextVar[str] = ContextVar("trace_id", default="-")

_EXTRA_FIELDS = (
    "trace_id",
    "src_lang",
    "tgt_lang",
    "domain",
    "engine",
    "cached",
    "tokens",
    "duration_ms",
    "endpoint",
    "status",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": getattr(record, "trace_id", trace_id_var.get()),
        }
        for field in _EXTRA_FIELDS:
            if field != "trace_id" and hasattr(record, field):
                payload[field] = getattr(record, field)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def setup_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # 降低第三方库噪音
    for noisy in ("uvicorn.access", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
