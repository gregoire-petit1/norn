"""Tests for memory models."""

from datetime import UTC, datetime

from norn.memory.models import DreamMetadata, MemoryConfig, SessionSummary


def test_memory_config_defaults():
    config = MemoryConfig()
    assert config.memory_dir.name == "memory"
    assert config.max_memory_lines == 200
    assert config.max_memory_bytes == 25_000
    assert config.dream_interval_hours == 24
    assert config.dream_min_sessions == 5


def test_memory_config_custom_dir():
    config = MemoryConfig(memory_dir="/tmp/test-memory")
    assert str(config.memory_dir) == "/tmp/test-memory"


def test_session_summary_creation():
    summary = SessionSummary(
        session_id="abc123",
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=UTC),
        ended_at=datetime(2026, 3, 31, 10, 30, tzinfo=UTC),
        user_messages=5,
        tool_calls=12,
        summary=(
            "User asked to refactor the auth module."
            " Extracted token validation into a separate service."
        ),
        topics=["auth", "refactoring"],
    )
    assert summary.session_id == "abc123"
    assert summary.user_messages == 5
    assert summary.duration_minutes == 30


def test_session_summary_duration_calculation():
    summary = SessionSummary(
        session_id="x",
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=UTC),
        ended_at=datetime(2026, 3, 31, 11, 15, tzinfo=UTC),
        user_messages=1,
        tool_calls=0,
        summary="Quick question.",
    )
    assert summary.duration_minutes == 75


def test_dream_metadata_creation():
    meta = DreamMetadata(
        dreamed_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        sessions_processed=5,
        files_written=["MEMORY.md", "topics/auth.md"],
        files_pruned=["daily/2026-03-25.md"],
        model_used="qwen2.5-coder:7b",
    )
    assert meta.sessions_processed == 5
    assert len(meta.files_written) == 2


def test_dream_metadata_defaults():
    meta = DreamMetadata(
        dreamed_at=datetime(2026, 3, 31, 12, 0, tzinfo=UTC),
        sessions_processed=0,
    )
    assert meta.files_written == []
    assert meta.files_pruned == []
    assert meta.model_used is None
