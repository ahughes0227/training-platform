"""Structured logging and optional OTLP export shared by all runtime processes."""

from __future__ import annotations

import contextvars
import json
import logging
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

_context: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar(
    "defect_log_context", default=None
)


class TokenBudgetExceeded(ValueError):
    """Raised when a model operation exceeds its configured token budget."""


@dataclass(frozen=True)
class TokenBudget:
    """Per-operation limits used by model adapters before accepting output."""

    max_prompt_tokens: int | None = None
    max_completion_tokens: int | None = None
    max_total_tokens: int | None = None

    @classmethod
    def from_env(cls, prefix: str = "DEFECT_AGENT_") -> TokenBudget | None:
        def read(name: str) -> int | None:
            value = os.getenv(prefix + name)
            return None if value in (None, "") else int(value)

        values = (
            read("MAX_PROMPT_TOKENS"),
            read("MAX_COMPLETION_TOKENS"),
            read("MAX_TOTAL_TOKENS"),
        )
        return cls(*values) if any(value is not None for value in values) else None

    def validate(self, telemetry: AgentTelemetry) -> AgentTelemetry:
        checks = (
            ("prompt", self.max_prompt_tokens, telemetry.prompt_tokens),
            ("completion", self.max_completion_tokens, telemetry.completion_tokens),
            ("total", self.max_total_tokens, telemetry.total_tokens),
        )
        for label, limit, value in checks:
            if limit is not None and (limit < 0 or value > limit):
                raise TokenBudgetExceeded(f"{label} token budget exceeded: {value} > {limit}")
        return telemetry


@dataclass
class AgentTelemetry:
    """Bounded usage counters for setup agents and API clients.

    The control plane records these counters as metadata; it never needs to
    retain prompts or responses, so token accounting does not make execution
    depend on a live model session.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    retries: int = 0
    context_tokens: int = 0
    repeated_context_tokens: int = 0
    context_limit: int | None = None

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def as_dict(self) -> dict[str, int | None]:
        result = asdict(self)
        result["total_tokens"] = self.total_tokens
        return result


def telemetry_from_response(
    response: Any, *, retries: int = 0, context_tokens: int = 0, context_limit: int | None = None
) -> AgentTelemetry:
    """Extract provider-neutral token usage from a LiteLLM/OpenAI response."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")
    if usage is None:
        usage = {}

    def value(*names: str) -> int:
        for name in names:
            item = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
            if item is not None:
                return int(item)
        return 0

    return AgentTelemetry(
        prompt_tokens=value("prompt_tokens", "input_tokens"),
        completion_tokens=value("completion_tokens", "output_tokens"),
        retries=retries,
        context_tokens=context_tokens,
        context_limit=context_limit,
    )


def bind_context(**identifiers: str) -> contextvars.Token[dict[str, str] | None]:
    """Attach traceable identifiers to logs in the current async context."""
    merged = {
        **(_context.get() or {}),
        **{key: value for key, value in identifiers.items() if value},
    }
    return _context.set(merged)


def record_agent_telemetry(
    *,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    retries: int = 0,
    context_tokens: int = 0,
    repeated_context_tokens: int = 0,
    context_limit: int | None = None,
) -> contextvars.Token[dict[str, str] | None]:
    """Bind token, retry, and context counters to subsequent structured logs."""
    values = AgentTelemetry(
        prompt_tokens,
        completion_tokens,
        retries,
        context_tokens,
        repeated_context_tokens,
        context_limit,
    )
    if (
        min(
            values.prompt_tokens,
            values.completion_tokens,
            values.retries,
            values.context_tokens,
            values.repeated_context_tokens,
        )
        < 0
    ):
        raise ValueError("agent telemetry counters cannot be negative")
    return bind_context(
        **{
            f"agent.{key}": str(value)
            for key, value in values.as_dict().items()
            if value is not None
        }
    )


record_token_usage = record_agent_telemetry
bind_agent_context = record_agent_telemetry


def record_retry(count: int = 1) -> contextvars.Token[dict[str, str] | None]:
    """Attach a retry count to the current operation's structured context."""
    if count < 0:
        raise ValueError("retry count cannot be negative")
    return bind_context(**{"agent.retries": str(count)})


def record_context_usage(
    tokens: int, limit: int | None = None
) -> contextvars.Token[dict[str, str] | None]:
    """Attach context-window usage without retaining prompt content."""
    if tokens < 0 or (limit is not None and limit < 0):
        raise ValueError("context counters cannot be negative")
    values = {"agent.context_tokens": str(tokens)}
    if limit is not None:
        values["agent.context_limit"] = str(limit)
    return bind_context(**values)


@contextmanager
def token_telemetry(**counters: int | None):
    """Temporarily attach usage counters to logs for one agent operation."""
    values = AgentTelemetry(
        prompt_tokens=int(counters.get("prompt_tokens", 0) or 0),
        completion_tokens=int(counters.get("completion_tokens", 0) or 0),
        retries=int(counters.get("retries", 0) or 0),
        context_tokens=int(counters.get("context_tokens", 0) or 0),
        repeated_context_tokens=int(counters.get("repeated_context_tokens", 0) or 0),
        context_limit=(
            int(limit_value) if (limit_value := counters.get("context_limit")) is not None else None
        ),
    )
    token = record_agent_telemetry(**asdict(values))
    log.info("agent.telemetry", extra={"agent_telemetry": values.as_dict()})
    try:
        yield values
    finally:
        reset_context(token)


@contextmanager
def response_telemetry(
    response: Any,
    *,
    retries: int = 0,
    context_tokens: int = 0,
    context_limit: int | None = None,
    prompt_tokens: int | None = None,
    budget: TokenBudget | None = None,
):
    """Bind usage extracted from a provider response for one operation."""
    values = telemetry_from_response(
        response, retries=retries, context_tokens=context_tokens, context_limit=context_limit
    )
    if prompt_tokens is not None:
        values.prompt_tokens = max(values.prompt_tokens, prompt_tokens)
    if budget is not None:
        budget.validate(values)
    token = record_agent_telemetry(**asdict(values))
    log.info("agent.telemetry", extra={"agent_telemetry": values.as_dict()})
    try:
        yield values
    finally:
        reset_context(token)


def reset_context(token: contextvars.Token[dict[str, str] | None]) -> None:
    _context.reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "severity": record.levelname,
            "service.name": os.getenv("OTEL_SERVICE_NAME", "defect-platform"),
            "message": record.getMessage(),
            **(_context.get() or {}),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for name in ("stage", "failure_code", "vertex_job_name", "image_digest"):
            if hasattr(record, name):
                payload[name] = getattr(record, name)
        agent_telemetry = getattr(record, "agent_telemetry", None)
        if agent_telemetry is not None:
            payload["agent_telemetry"] = agent_telemetry
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
