"""Tests for structlog processors."""

from __future__ import annotations

import pytest

from norn.observability.logger import new_session, session_id_var
from norn.observability.processors import (
    add_session_id,
    add_timestamp_iso,
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
