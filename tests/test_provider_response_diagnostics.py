from __future__ import annotations

import json
from types import SimpleNamespace

from agent import provider_response_diagnostics as diagnostics


def _read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_non_streaming_response_capture_is_minimal_and_redacted(tmp_path, monkeypatch):
    output = tmp_path / "provider.jsonl"
    monkeypatch.setenv("HERMES_PROVIDER_RESPONSE_DIAGNOSTICS_PATH", str(output))

    message = SimpleNamespace(
        content="token=super-secret visible text",
        tool_calls=[
            SimpleNamespace(
                id="call-1",
                type="function",
                function=SimpleNamespace(name="vision_analyze", arguments='{"image_url":"/tmp/x.jpg"}'),
            )
        ],
        role="assistant",
        model_extra={"provider_hint": "custom"},
    )
    choice = SimpleNamespace(message=message, finish_reason="tool_calls")
    response = SimpleNamespace(choices=[choice], model="MiniMaxAI/MiniMax-M2.7")

    diagnostics._record_response(response, "MiniMaxAI/MiniMax-M2.7")

    [record] = _read_jsonl(output)
    assert record["kind"] == "non_streaming_response"
    assert record["finish_reason"] == "tool_calls"
    assert "super-secret" not in record["content"]
    assert "***REDACTED***" in record["content"]
    assert record["tool_calls"][0]["function"]["name"] == "vision_analyze"
    assert record["unknown_message_fields"] == ["provider_hint"]
    assert "messages" not in record
    assert "headers" not in record


def test_stream_chunk_capture_records_content_tool_calls_and_finish_reason(tmp_path, monkeypatch):
    output = tmp_path / "provider.jsonl"
    monkeypatch.setenv("HERMES_PROVIDER_RESPONSE_DIAGNOSTICS_PATH", str(output))

    delta = SimpleNamespace(
        content="[tool vision_analyze]",
        tool_calls=None,
        role="assistant",
        model_extra={"vendor_field": "value"},
    )
    choice = SimpleNamespace(delta=delta, finish_reason="stop")
    chunk = SimpleNamespace(
        choices=[choice],
        model="MiniMaxAI/MiniMax-M2.7",
        model_extra={"vendor_chunk_field": "value"},
    )

    diagnostics._record_chunk(chunk, "fallback-model")

    [record] = _read_jsonl(output)
    assert record["kind"] == "stream_chunk"
    assert record["model"] == "MiniMaxAI/MiniMax-M2.7"
    assert record["delta_content"] == "[tool vision_analyze]"
    assert record["delta_tool_calls"] is None
    assert record["finish_reason"] == "stop"
    assert record["unknown_delta_fields"] == ["vendor_field"]
    assert record["unknown_chunk_fields"] == ["vendor_chunk_field"]


def test_write_failures_never_escape(monkeypatch):
    monkeypatch.setenv(
        "HERMES_PROVIDER_RESPONSE_DIAGNOSTICS_PATH",
        "/proc/does-not-exist/provider.jsonl",
    )

    diagnostics._write({"kind": "test", "content": "safe"})
