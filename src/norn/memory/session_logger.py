"""Session logger: records session summaries to daily logs."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.memory.models import SessionSummary
    from norn.memory.store import MemoryStore


class SessionLogger:
    """Logs completed session summaries to daily files."""

    def __init__(self, store: MemoryStore) -> None:
        self._store = store
        self._session_ids: list[str] = []

    def log_session(self, summary: SessionSummary) -> None:
        """Log a session summary to the daily log."""
        date_str = summary.started_at.strftime("%Y-%m-%d")
        topics_str = ", ".join(summary.topics) if summary.topics else "none"
        entry = (
            f"### Session {summary.session_id}\n"
            f"- **Duration**: {summary.duration_minutes} min | "
            f"{summary.user_messages} messages | "
            f"{summary.tool_calls} tool calls\n"
            f"- **Topics**: {topics_str}\n"
            f"- **Summary**: {summary.summary}\n"
        )
        self._store.append_daily(date_str, entry)
        self._session_ids.append(summary.session_id)

    def session_count(self) -> int:
        """Return the number of sessions logged in this process lifetime."""
        return len(self._session_ids)
