"""Tests for the session logger."""

from datetime import UTC, datetime

import pytest

from norn.memory.models import MemoryConfig, SessionSummary
from norn.memory.session_logger import SessionLogger
from norn.memory.store import MemoryStore


@pytest.fixture
def store(tmp_path):
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    s = MemoryStore(config)
    s.ensure_dirs()
    return s


@pytest.fixture
def logger(store):
    return SessionLogger(store)


def _make_summary(
    session_id: str = "test-001",
    summary: str = "User refactored the auth module.",
    topics: list[str] | None = None,
) -> SessionSummary:
    return SessionSummary(
        session_id=session_id,
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=UTC),
        ended_at=datetime(2026, 3, 31, 10, 30, tzinfo=UTC),
        user_messages=5,
        tool_calls=12,
        summary=summary,
        topics=topics or ["auth"],
    )


def test_log_session_appends_to_daily(logger, store):
    """Logging a session appends to the correct daily log."""
    summary = _make_summary()
    logger.log_session(summary)
    content = store.read_daily("2026-03-31")
    assert content is not None
    assert "test-001" in content
    assert "refactored the auth" in content


def test_log_session_includes_metadata(logger, store):
    """Session log entry includes messages, tools, duration."""
    summary = _make_summary()
    logger.log_session(summary)
    content = store.read_daily("2026-03-31")
    assert "5 messages" in content
    assert "12 tool calls" in content
    assert "30 min" in content


def test_log_session_includes_topics(logger, store):
    """Session log entry includes topic tags."""
    summary = _make_summary(topics=["auth", "refactoring"])
    logger.log_session(summary)
    content = store.read_daily("2026-03-31")
    assert "auth" in content
    assert "refactoring" in content


def test_log_multiple_sessions(logger, store):
    """Multiple sessions on the same day are all recorded."""
    logger.log_session(_make_summary(session_id="s1", summary="Did A"))
    logger.log_session(_make_summary(session_id="s2", summary="Did B"))
    content = store.read_daily("2026-03-31")
    assert "s1" in content
    assert "s2" in content
    assert "Did A" in content
    assert "Did B" in content


def test_session_count_since(logger):
    """Count sessions since a given timestamp."""
    logger.log_session(_make_summary(session_id="s1"))
    logger.log_session(_make_summary(session_id="s2"))
    logger.log_session(_make_summary(session_id="s3"))
    count = logger.session_count()
    assert count == 3


def test_session_count_empty(logger):
    assert logger.session_count() == 0
