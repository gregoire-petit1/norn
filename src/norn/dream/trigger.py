"""Dream trigger: three-gate system for deciding when to consolidate memory."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from filelock import FileLock, Timeout

_LAST_DREAM_FILE = ".last_dream"
_DREAM_LOCK_FILE = ".dream.lock"
_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


class DreamTrigger:
    """Evaluates three gates to decide if a dream should run."""

    def __init__(
        self,
        memory_dir: Path,
        interval_hours: int = 24,
        min_sessions: int = 5,
    ) -> None:
        self._memory_dir = Path(memory_dir)
        self._interval_hours = interval_hours
        self._min_sessions = min_sessions
        self._lock_file = self._memory_dir / _DREAM_LOCK_FILE
        self._last_dream_file = self._memory_dir / _LAST_DREAM_FILE
        self._file_lock: FileLock | None = None

    # --- Time gate ---

    def check_time_gate(self) -> bool:
        """Check if enough time has passed since the last dream."""
        last_dream = self._read_last_dream_time()
        if last_dream is None:
            return True
        threshold = datetime.now(tz=UTC) - timedelta(hours=self._interval_hours)
        return last_dream <= threshold

    def _read_last_dream_time(self) -> datetime | None:
        """Read the last dream timestamp from file."""
        if not self._last_dream_file.exists():
            return None
        text = self._last_dream_file.read_text().strip()
        if not text:
            return None
        return datetime.strptime(text, _TIMESTAMP_FORMAT)

    def write_last_dream_time(self, dt: datetime) -> None:
        """Write the last dream timestamp to file."""
        self._last_dream_file.write_text(dt.strftime(_TIMESTAMP_FORMAT))

    # --- Session gate ---

    def check_session_gate(self, session_count: int) -> bool:
        """Check if enough sessions have accumulated."""
        return session_count >= self._min_sessions

    # --- Lock gate ---

    def try_acquire_lock(self) -> bool:
        """Try to acquire the dream lock (non-blocking)."""
        self._file_lock = FileLock(self._lock_file)
        try:
            self._file_lock.acquire(timeout=0)
            return True
        except Timeout:
            self._file_lock = None
            return False

    def release_lock(self) -> None:
        """Release the dream lock."""
        if self._file_lock is not None:
            self._file_lock.release()
            self._file_lock = None

    # --- Combined check ---

    def should_dream(self, session_count: int) -> bool:
        """Check all three gates. Acquires lock if all pass (caller must release)."""
        if not self.check_time_gate():
            return False
        if not self.check_session_gate(session_count):
            return False
        return self.try_acquire_lock()
