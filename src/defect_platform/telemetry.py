"""Structured logging and optional OTLP export shared by all runtime processes."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from datetime import datetime, timezone
from typing import Any


_context: contextvars.ContextVar[dict[str, str]] = contextvars.ContextVar(
    "defect_log_context", default={}
)


def bind_context(**identifiers: str) -> contextvars.Token[dict[str, str]]:
    """Attach traceable identifiers to logs in the current async context."""
    merged = {**_context.get(), **{key: value for key, value in identifiers.items() if value}}
    return _context.set(merged)


def reset_context(token: contextvars.Token[dict[str, str]]) -> None:
    _context.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "severity": record.levelname,
            "service.name": os.getenv("OTEL_SERVICE_NAME", "defect-platform"),
            "message": record.getMessage(),
            **_context.get(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for name in ("stage", "failure_code", "vertex_job_name", "image_digest"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        return json.dumps(payload, sort_keys=True, default=str)


def configure_logging(level: str = "INFO", otlp_endpoint: str | None = None) -> None:
    """Always emit JSON to stdout; add OTLP log export only when configured.

    The collector can forward OTLP logs to Loki's native OTLP endpoint. Run IDs
    remain structured metadata, avoiding high-cardinality Loki index labels.
    """
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level.upper())
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())
    root.addHandler(stream)
    endpoint = otlp_endpoint or os.getenv("DEFECT_OTLP_ENDPOINT")
    if not endpoint:
        return
    try:
        from opentelemetry import _logs
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        provider = LoggerProvider(resource=Resource.create({"service.name": "defect-platform"}))
        provider.add_log_record_processor(
            BatchLogRecordProcessor(OTLPLogExporter(endpoint=endpoint))
        )
        _logs.set_logger_provider(provider)
        root.addHandler(LoggingHandler(level=root.level, logger_provider=provider))
    except ImportError as exc:
        root.warning("OTLP endpoint configured but telemetry extra is missing: %s", exc)
