"""Memory system models for Norn."""

from __future__ import annotations

from datetime import datetime  # noqa: TCH003 - Pydantic needs this at runtime
from pathlib import Path

from pydantic import BaseModel, computed_field


class MemoryConfig(BaseModel):
    """Configuration for the memory system."""

    memory_dir: Path = Path.home() / ".norn" / "memory"
    max_memory_lines: int = 200
    max_memory_bytes: int = 25_000
    dream_interval_hours: int = 24
    dream_min_sessions: int = 5


class SessionSummary(BaseModel):
    """Summary of a completed agent session."""

    session_id: str
    started_at: datetime
    ended_at: datetime
    user_messages: int
    tool_calls: int
    summary: str
    topics: list[str] = []

    @computed_field
    @property
    def duration_minutes(self) -> int:
        """Duration in minutes."""
        delta = self.ended_at - self.started_at
        return int(delta.total_seconds() / 60)


class DreamMetadata(BaseModel):
    """Metadata about a completed dream consolidation."""

    dreamed_at: datetime
    sessions_processed: int
    files_written: list[str] = []
    files_pruned: list[str] = []
    model_used: str | None = None
