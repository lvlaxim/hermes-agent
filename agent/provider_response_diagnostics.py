"""Opt-in diagnostics for OpenAI-compatible Chat Completions responses.

The hook is disabled unless ``HERMES_PROVIDER_RESPONSE_DIAGNOSTICS`` is truthy.
When enabled, it wraps the OpenAI SDK's synchronous ``Completions.create``
method and writes a minimal, redacted JSONL record for non-streaming responses
and streaming chunks. Request messages, headers and credentials are never
written.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_ENV_FLAG = "HERMES_PROVIDER_RESPONSE_DIAGNOSTICS"
_ENV_PATH = "HERMES_PROVIDER_RESPONSE_DIAGNOSTICS_PATH"
_DEFAULT_PATH = "/opt/data/sessions/provider_response_diagnostics.jsonl"
_MAX_TEXT = 4000
_MAX_VALUE = 8000
_LOCK = threading.Lock()
_INSTALLED = False

_SECRET_PATTERNS = (
    re.compile(
        r"(?i)(api[_-]?key|token|secret|password|authorization)"
        r"([\"'=:\s]+)([^\s,;\"'}]+)"
    ),
    re.compile(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+"),
)

_KNOWN_MESSAGE_FIELDS = {
    "content",
    "refusal",
    "role",
    "function_call",
    "tool_calls",
    "audio",
    "annotations",
}
_KNOWN_DELTA_FIELDS = {
    "content",
    "refusal",
    "role",
    "function_call",
    "tool_calls",
    "audio",
}


def _enabled() -> bool:
    return os.getenv(_ENV_FLAG, "").strip().lower() in {"1", "true", "yes", "on"}


def _redact(text: str) -> str:
    value = text
    value = _SECRET_PATTERNS[0].sub(r"\1\2***REDACTED***", value)
    value = _SECRET_PATTERNS[1].sub(r"\1***REDACTED***", value)
    return value


def _truncate(value: Any, limit: int = _MAX_VALUE) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        redacted = _redact(value)
        return redacted if len(redacted) <= limit else redacted[:limit] + "...[truncated]"
    if isinstance(value, list):
        return [_truncate(item, limit) for item in value[:50]]
    if isinstance(value, tuple):
        return [_truncate(item, limit) for item in value[:50]]
    if isinstance(value, dict):
        return {
            str(key): _truncate(item, limit)
            for key, item in list(value.items())[:100]
            if not _looks_sensitive_key(str(key))
        }
    if hasattr(value, "model_dump"):
        try:
            return _truncate(value.model_dump(), limit)
        except Exception:
            pass
    namespace = getattr(value, "__dict__", None)
    if isinstance(namespace, dict):
        return _truncate(namespace, limit)
    return _truncate(str(value), limit)


def _looks_sensitive_key(key: str) -> bool:
    lowered = key.lower()
    return any(marker in lowered for marker in ("key", "token", "secret", "password", "authorization", "header"))


def _model_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        try:
            dumped = value.model_dump()
            if isinstance(dumped, dict):
                return dumped
        except Exception:
            pass
    if isinstance(value, dict):
        return value
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        return dict(model_extra)
    return {}


def _unknown_fields(value: Any, known: set[str]) -> list[str]:
    fields = set(_model_dict(value))
    model_extra = getattr(value, "model_extra", None)
    if isinstance(model_extra, dict):
        fields.update(model_extra)
    return sorted(field for field in fields if field not in known and not _looks_sensitive_key(field))


def _tool_calls(value: Any) -> Any:
    calls = getattr(value, "tool_calls", None)
    return _truncate(calls)


def _base_record(kind: str, model: Any) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "kind": kind,
        "model": _truncate(model, 512),
    }


def _write(record: dict[str, Any]) -> None:
    try:
        path = Path(os.getenv(_ENV_PATH, _DEFAULT_PATH))
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(_truncate(record), ensure_ascii=False, separators=(",", ":"))
        with _LOCK:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception as exc:  # diagnostics must never affect the request path
        logger.debug("Provider response diagnostics write failed: %s", exc)


def _record_response(response: Any, model: Any) -> None:
    try:
        choices = getattr(response, "choices", None) or []
        choice = choices[0] if choices else None
        message = getattr(choice, "message", None) if choice is not None else None
        record = _base_record("non_streaming_response", model)
        record.update(
            {
                "finish_reason": _truncate(getattr(choice, "finish_reason", None), 256),
                "content": _truncate(getattr(message, "content", None), _MAX_TEXT),
                "tool_calls": _tool_calls(message),
                "unknown_message_fields": _unknown_fields(message, _KNOWN_MESSAGE_FIELDS),
                "unknown_response_fields": _unknown_fields(response, {"id", "choices", "created", "model", "object", "service_tier", "system_fingerprint", "usage"}),
            }
        )
        _write(record)
    except Exception as exc:
        logger.debug("Provider response diagnostics capture failed: %s", exc)


def _record_chunk(chunk: Any, model: Any) -> None:
    try:
        choices = getattr(chunk, "choices", None) or []
        choice = choices[0] if choices else None
        delta = getattr(choice, "delta", None) if choice is not None else None
        record = _base_record("stream_chunk", getattr(chunk, "model", None) or model)
        record.update(
            {
                "finish_reason": _truncate(getattr(choice, "finish_reason", None), 256),
                "delta_content": _truncate(getattr(delta, "content", None), _MAX_TEXT),
                "delta_tool_calls": _tool_calls(delta),
                "unknown_delta_fields": _unknown_fields(delta, _KNOWN_DELTA_FIELDS),
                "unknown_chunk_fields": _unknown_fields(chunk, {"id", "choices", "created", "model", "object", "service_tier", "system_fingerprint", "usage"}),
            }
        )
        _write(record)
    except Exception as exc:
        logger.debug("Provider stream diagnostics capture failed: %s", exc)


class _StreamProxy:
    def __init__(self, stream: Any, model: Any):
        self._stream = stream
        self._model = model

    def __iter__(self):
        return self

    def __next__(self):
        chunk = next(self._stream)
        _record_chunk(chunk, self._model)
        return chunk

    def __enter__(self):
        enter = getattr(self._stream, "__enter__", None)
        if callable(enter):
            enter()
        return self

    def __exit__(self, exc_type, exc, tb):
        exit_method = getattr(self._stream, "__exit__", None)
        if callable(exit_method):
            return exit_method(exc_type, exc, tb)
        close = getattr(self._stream, "close", None)
        if callable(close):
            close()
        return False

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def install_provider_response_diagnostics() -> None:
    """Install the OpenAI Chat Completions response hook once, when enabled."""
    global _INSTALLED
    if _INSTALLED or not _enabled():
        return

    try:
        from openai.resources.chat.completions.completions import Completions

        original_create = Completions.create

        def wrapped_create(self, *args, **kwargs):
            response = original_create(self, *args, **kwargs)
            model = kwargs.get("model")
            if kwargs.get("stream"):
                return _StreamProxy(response, model)
            _record_response(response, model)
            return response

        Completions.create = wrapped_create
        _INSTALLED = True
        logger.info("Provider response diagnostics enabled")
    except Exception as exc:
        logger.debug("Provider response diagnostics installation failed: %s", exc)


install_provider_response_diagnostics()
