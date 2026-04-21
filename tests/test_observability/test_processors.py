"""Tests for structlog processors."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from norn.observability.logger import new_session, session_id_var
from norn.observability.processors import (
    add_session_id,
    add_timestamp_iso,
    make_cost_processor,
    redact_secrets,
)


@pytest.fixture(autouse=True)
def reset_session():
    token = session_id_var.set(None)
    yield
    session_id_var.reset(token)


def test_add_session_id_when_set():
    sid = new_session()
    out = add_session_id(None, "info", {"event": "test"})
    assert out["session_id"] == sid


def test_add_session_id_absent_when_unset():
    out = add_session_id(None, "info", {"event": "test"})
    assert "session_id" not in out


def test_add_timestamp_iso_format():
    out = add_timestamp_iso(None, "info", {"event": "test"})
    assert "timestamp" in out
    # ISO-8601 UTC, e.g. "2026-04-21T14:23:01.123456Z"
    assert out["timestamp"].endswith("Z")
    assert "T" in out["timestamp"]


def test_redact_secrets_strips_sensitive_fields():
    event = {
        "event": "llm.complete",
        "api_key": "sk-secret",
        "authorization": "Bearer xyz",
        "token": "abc",
        "password": "hunter2",
        "model": "gpt-4",
    }
    out = redact_secrets(
        None, "info", event, redact_keys=["api_key", "authorization", "token", "password"]
    )
    assert out["api_key"] == "[REDACTED]"
    assert out["authorization"] == "[REDACTED]"
    assert out["token"] == "[REDACTED]"
    assert out["password"] == "[REDACTED]"
    assert out["model"] == "gpt-4"  # untouched
    assert out["event"] == "llm.complete"


def test_redact_secrets_case_insensitive():
    event = {"API_KEY": "secret", "Authorization": "bearer"}
    out = redact_secrets(None, "info", event, redact_keys=["api_key", "authorization"])
    assert out["API_KEY"] == "[REDACTED]"
    assert out["Authorization"] == "[REDACTED]"


def test_redact_does_not_clobber_token_count_fields():
    """Regression: default redact_keys must not match prompt_tokens/completion_tokens.

    Before the exact-match fix, 'token' as a redact key used substring
    matching, which clobbered LLM usage metrics.
    """
    event = {
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "total_tokens": 59,
        "token": "secret-value",
    }
    out = redact_secrets(
        None,
        "info",
        event,
        redact_keys=["token", "api_key", "authorization", "password"],
    )
    assert out["prompt_tokens"] == 42
    assert out["completion_tokens"] == 17
    assert out["total_tokens"] == 59
    assert out["token"] == "[REDACTED]"  # exact match still redacted


def test_cost_processor_disabled_adds_nothing():
    processor = make_cost_processor(enabled=False)
    out = processor(
        None,
        "info",
        {
            "event": "llm.complete",
            "model": "gpt-4",
            "prompt_tokens": 10,
            "completion_tokens": 5,
        },
    )
    assert "cost_usd" not in out


def test_cost_processor_ignores_non_llm_events():
    processor = make_cost_processor(enabled=True)
    out = processor(None, "info", {"event": "tool.call", "tool_name": "read_file"})
    assert "cost_usd" not in out


def test_cost_processor_happy_path():
    processor = make_cost_processor(enabled=True)
    event = {
        "event": "llm.complete",
        "model": "gpt-4",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    # cost_per_token returns (prompt_cost, completion_cost); processor sums them.
    with patch("litellm.cost_per_token", return_value=(0.0001, 0.00002345678)):
        out = processor(None, "info", event)
    assert out["cost_usd"] == 0.000123  # (0.0001 + 0.00002345678) rounded to 6 decimals


def test_cost_processor_fail_open_on_exception():
    processor = make_cost_processor(enabled=True)
    event = {
        "event": "llm.complete",
        "model": "unknown-model",
        "prompt_tokens": 10,
        "completion_tokens": 5,
    }
    with patch("litellm.cost_per_token", side_effect=RuntimeError("unknown model")):
        out = processor(None, "info", event)
    assert out["cost_usd"] is None
