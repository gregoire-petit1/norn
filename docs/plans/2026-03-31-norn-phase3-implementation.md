# Norn Phase 3: Dream System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a persistent memory system with automatic consolidation ("dreaming"), inspired by Claude Code's `autoDream`. The system stores session summaries, maintains a curated `MEMORY.md` index, and periodically consolidates memories via a 3-gate trigger + 4-phase consolidation pipeline.

**Architecture:** The memory layer lives at `~/.norn/memory/` as plain markdown files (human-readable, git-friendly, no DB). A `MemoryStore` manages reading/writing memory files. A `SessionLogger` appends session summaries to daily logs. A `DreamTrigger` evaluates 3 gates (time, session count, file lock) to decide if dreaming should occur. A `DreamEngine` orchestrates the 4-phase consolidation (Orient, Gather, Consolidate, Prune) using an LLM call. The `AgentLoop` injects `MEMORY.md` into the system prompt and triggers dreams as fire-and-forget background tasks at session end.

**Tech Stack:** Python 3.11+, Pydantic v2, aiofiles, pytest, ruff, filelock (new dependency)

---

## Dependencies

Add `filelock>=3.15` to `pyproject.toml` dependencies for the lock gate:

```toml
dependencies = [
    ...,
    "filelock>=3.15",
]
```

Run: `uv pip install -e ".[dev]"` after updating.

---

## Task 0: Memory Models and Directory Structure

**Files:**
- Create: `src/norn/memory/__init__.py`
- Create: `src/norn/memory/models.py`
- Create: `tests/test_memory/__init__.py`
- Create: `tests/test_memory/test_models.py`

**Step 1: Create test directories**

```bash
mkdir -p tests/test_memory
touch tests/test_memory/__init__.py
mkdir -p src/norn/memory
touch src/norn/memory/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_memory/test_models.py
"""Tests for memory models."""

from datetime import datetime, timezone

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
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 31, 10, 30, tzinfo=timezone.utc),
        user_messages=5,
        tool_calls=12,
        summary="User asked to refactor the auth module. Extracted token validation into a separate service.",
        topics=["auth", "refactoring"],
    )
    assert summary.session_id == "abc123"
    assert summary.user_messages == 5
    assert summary.duration_minutes == 30


def test_session_summary_duration_calculation():
    summary = SessionSummary(
        session_id="x",
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 31, 11, 15, tzinfo=timezone.utc),
        user_messages=1,
        tool_calls=0,
        summary="Quick question.",
    )
    assert summary.duration_minutes == 75


def test_dream_metadata_creation():
    meta = DreamMetadata(
        dreamed_at=datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        sessions_processed=5,
        files_written=["MEMORY.md", "topics/auth.md"],
        files_pruned=["daily/2026-03-25.md"],
        model_used="qwen2.5-coder:7b",
    )
    assert meta.sessions_processed == 5
    assert len(meta.files_written) == 2


def test_dream_metadata_defaults():
    meta = DreamMetadata(
        dreamed_at=datetime(2026, 3, 31, 12, 0, tzinfo=timezone.utc),
        sessions_processed=0,
    )
    assert meta.files_written == []
    assert meta.files_pruned == []
    assert meta.model_used is None
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory/test_models.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'norn.memory.models'`

**Step 4: Write minimal implementation**

```python
# src/norn/memory/models.py
"""Memory system models for Norn."""

from __future__ import annotations

from datetime import datetime
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
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory/test_models.py -v`
Expected: 6 PASSED

**Step 6: Run ruff**

Run: `uv run ruff check src/norn/memory/models.py`

**Step 7: Commit**

```bash
git add src/norn/memory/ tests/test_memory/
git commit -m "feat(memory): add memory models and session summary types"
```

---

## Task 1: MemoryStore -- Read/Write Memory Files

**Files:**
- Create: `src/norn/memory/store.py`
- Create: `tests/test_memory/test_store.py`

The MemoryStore handles all filesystem operations for the memory directory. It creates the directory structure on init, reads/writes `MEMORY.md`, manages topic files, and appends to daily logs.

**Step 1: Write the failing tests**

```python
# tests/test_memory/test_store.py
"""Tests for the memory store."""

from pathlib import Path

import pytest

from norn.memory.models import MemoryConfig
from norn.memory.store import MemoryStore


@pytest.fixture
def memory_dir(tmp_path):
    """Create a temporary memory directory."""
    return tmp_path / "memory"


@pytest.fixture
def store(memory_dir):
    """Create a MemoryStore with a temp directory."""
    config = MemoryConfig(memory_dir=memory_dir)
    return MemoryStore(config)


class TestInit:
    def test_ensures_directory_structure(self, store, memory_dir):
        """MemoryStore should create the directory tree on init."""
        store.ensure_dirs()
        assert (memory_dir / "topics").is_dir()
        assert (memory_dir / "daily").is_dir()

    def test_creates_empty_memory_if_missing(self, store, memory_dir):
        """If MEMORY.md doesn't exist, create with header."""
        store.ensure_dirs()
        memory_file = memory_dir / "MEMORY.md"
        assert memory_file.exists()
        content = memory_file.read_text()
        assert "# Norn Memory" in content


class TestReadWrite:
    def test_read_memory(self, store, memory_dir):
        """Read MEMORY.md content."""
        store.ensure_dirs()
        content = store.read_memory()
        assert "# Norn Memory" in content

    def test_write_memory(self, store, memory_dir):
        """Write MEMORY.md content."""
        store.ensure_dirs()
        store.write_memory("# Norn Memory\n\n- User prefers dark mode\n")
        content = store.read_memory()
        assert "dark mode" in content

    def test_read_topic(self, store, memory_dir):
        """Read a topic file."""
        store.ensure_dirs()
        topic_path = memory_dir / "topics" / "auth.md"
        topic_path.write_text("# Auth\n\nUsing JWT tokens.\n")
        content = store.read_topic("auth")
        assert "JWT" in content

    def test_read_nonexistent_topic_returns_none(self, store):
        """Reading a topic that doesn't exist returns None."""
        store.ensure_dirs()
        assert store.read_topic("nonexistent") is None

    def test_write_topic(self, store, memory_dir):
        """Write a topic file."""
        store.ensure_dirs()
        store.write_topic("auth", "# Auth\n\nMigrated to OAuth2.\n")
        content = (memory_dir / "topics" / "auth.md").read_text()
        assert "OAuth2" in content

    def test_list_topics(self, store, memory_dir):
        """List all topic files."""
        store.ensure_dirs()
        (memory_dir / "topics" / "auth.md").write_text("x")
        (memory_dir / "topics" / "ml.md").write_text("y")
        topics = store.list_topics()
        assert set(topics) == {"auth", "ml"}

    def test_list_topics_empty(self, store):
        """No topics yet."""
        store.ensure_dirs()
        assert store.list_topics() == []


class TestDailyLogs:
    def test_append_daily_log(self, store, memory_dir):
        """Append to today's daily log."""
        store.ensure_dirs()
        store.append_daily("2026-03-31", "Session abc123: refactored auth module")
        log_path = memory_dir / "daily" / "2026-03-31.md"
        assert log_path.exists()
        content = log_path.read_text()
        assert "refactored auth" in content

    def test_append_daily_log_multiple(self, store, memory_dir):
        """Multiple appends to the same day."""
        store.ensure_dirs()
        store.append_daily("2026-03-31", "Session 1: did X")
        store.append_daily("2026-03-31", "Session 2: did Y")
        content = (memory_dir / "daily" / "2026-03-31.md").read_text()
        assert "Session 1" in content
        assert "Session 2" in content

    def test_read_daily_log(self, store, memory_dir):
        """Read a specific daily log."""
        store.ensure_dirs()
        store.append_daily("2026-03-31", "did stuff")
        content = store.read_daily("2026-03-31")
        assert "did stuff" in content

    def test_read_nonexistent_daily_returns_none(self, store):
        """Reading a daily log that doesn't exist returns None."""
        store.ensure_dirs()
        assert store.read_daily("2020-01-01") is None

    def test_list_daily_logs(self, store, memory_dir):
        """List daily logs sorted by date (most recent first)."""
        store.ensure_dirs()
        store.append_daily("2026-03-30", "day 1")
        store.append_daily("2026-03-31", "day 2")
        store.append_daily("2026-03-29", "day 0")
        dates = store.list_daily_logs()
        assert dates == ["2026-03-31", "2026-03-30", "2026-03-29"]


class TestMemorySize:
    def test_memory_line_count(self, store):
        """Count lines in MEMORY.md."""
        store.ensure_dirs()
        store.write_memory("line1\nline2\nline3\n")
        assert store.memory_line_count() == 3

    def test_memory_byte_size(self, store):
        """Get byte size of MEMORY.md."""
        store.ensure_dirs()
        content = "hello world"
        store.write_memory(content)
        assert store.memory_byte_size() == len(content.encode())
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory/test_store.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/memory/store.py
"""Memory store: manages reading and writing memory files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.memory.models import MemoryConfig

_MEMORY_HEADER = """# Norn Memory

> Auto-maintained by the dream system. Do not edit manually unless you know what you're doing.

"""


class MemoryStore:
    """Filesystem-backed memory store for Norn."""

    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._memory_dir = Path(config.memory_dir)
        self._memory_file = self._memory_dir / "MEMORY.md"
        self._topics_dir = self._memory_dir / "topics"
        self._daily_dir = self._memory_dir / "daily"

    def ensure_dirs(self) -> None:
        """Create directory structure and default files if they don't exist."""
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self._topics_dir.mkdir(exist_ok=True)
        self._daily_dir.mkdir(exist_ok=True)

        if not self._memory_file.exists():
            self._memory_file.write_text(_MEMORY_HEADER)

    # --- MEMORY.md ---

    def read_memory(self) -> str:
        """Read the main MEMORY.md file."""
        if not self._memory_file.exists():
            return ""
        return self._memory_file.read_text()

    def write_memory(self, content: str) -> None:
        """Write (overwrite) the main MEMORY.md file."""
        self._memory_file.write_text(content)

    def memory_line_count(self) -> int:
        """Count non-empty lines in MEMORY.md."""
        content = self.read_memory()
        if not content:
            return 0
        return len(content.strip().split("\n"))

    def memory_byte_size(self) -> int:
        """Get byte size of MEMORY.md."""
        content = self.read_memory()
        return len(content.encode())

    # --- Topic files ---

    def read_topic(self, name: str) -> str | None:
        """Read a topic file. Returns None if it doesn't exist."""
        path = self._topics_dir / f"{name}.md"
        if not path.exists():
            return None
        return path.read_text()

    def write_topic(self, name: str, content: str) -> None:
        """Write (overwrite) a topic file."""
        path = self._topics_dir / f"{name}.md"
        path.write_text(content)

    def list_topics(self) -> list[str]:
        """List all topic names (without .md extension)."""
        if not self._topics_dir.exists():
            return []
        return sorted(p.stem for p in self._topics_dir.glob("*.md"))

    # --- Daily logs ---

    def append_daily(self, date_str: str, entry: str) -> None:
        """Append an entry to a daily log file."""
        path = self._daily_dir / f"{date_str}.md"
        with open(path, "a") as f:
            f.write(entry + "\n\n")

    def read_daily(self, date_str: str) -> str | None:
        """Read a daily log. Returns None if it doesn't exist."""
        path = self._daily_dir / f"{date_str}.md"
        if not path.exists():
            return None
        return path.read_text()

    def list_daily_logs(self) -> list[str]:
        """List daily log dates, most recent first."""
        if not self._daily_dir.exists():
            return []
        dates = sorted((p.stem for p in self._daily_dir.glob("*.md")), reverse=True)
        return dates
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory/test_store.py -v`
Expected: 16 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/memory/store.py`

**Step 6: Commit**

```bash
git add src/norn/memory/store.py tests/test_memory/test_store.py
git commit -m "feat(memory): add filesystem-backed memory store with topics and daily logs"
```

---

## Task 2: SessionLogger -- Record Session Summaries

**Files:**
- Create: `src/norn/memory/session_logger.py`
- Create: `tests/test_memory/test_session_logger.py`

The SessionLogger wraps the MemoryStore to append structured session summaries to daily logs, and to save session metadata for dream processing.

**Step 1: Write the failing tests**

```python
# tests/test_memory/test_session_logger.py
"""Tests for the session logger."""

from datetime import datetime, timezone
from pathlib import Path

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
        started_at=datetime(2026, 3, 31, 10, 0, tzinfo=timezone.utc),
        ended_at=datetime(2026, 3, 31, 10, 30, tzinfo=timezone.utc),
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_memory/test_session_logger.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/memory/session_logger.py
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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_memory/test_session_logger.py -v`
Expected: 6 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/memory/session_logger.py`

**Step 6: Commit**

```bash
git add src/norn/memory/session_logger.py tests/test_memory/test_session_logger.py
git commit -m "feat(memory): add session logger for recording session summaries to daily logs"
```

---

## Task 3: DreamTrigger -- Three-Gate Decision

**Files:**
- Create: `src/norn/dream/__init__.py`
- Create: `src/norn/dream/trigger.py`
- Create: `tests/test_dream/__init__.py`
- Create: `tests/test_dream/test_trigger.py`

All three gates must pass for a dream to run:
1. **Time gate**: >= N hours since last dream (default 24)
2. **Session gate**: >= N sessions since last dream (default 5)
3. **Lock gate**: file lock acquired (no concurrent dream)

**Step 1: Create directories**

```bash
mkdir -p src/norn/dream tests/test_dream
touch src/norn/dream/__init__.py tests/test_dream/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_dream/test_trigger.py
"""Tests for the dream trigger (3-gate system)."""

from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from norn.dream.trigger import DreamTrigger


@pytest.fixture
def memory_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def trigger(memory_dir):
    return DreamTrigger(
        memory_dir=memory_dir,
        interval_hours=24,
        min_sessions=5,
    )


class TestTimeGate:
    def test_passes_when_no_previous_dream(self, trigger):
        """First dream ever: time gate passes."""
        assert trigger.check_time_gate() is True

    def test_passes_when_enough_time(self, trigger, memory_dir):
        """Enough time has passed since last dream."""
        # Write a dream timestamp 25 hours ago
        last_dream = datetime.now(tz=timezone.utc) - timedelta(hours=25)
        trigger.write_last_dream_time(last_dream)
        assert trigger.check_time_gate() is True

    def test_fails_when_too_recent(self, trigger, memory_dir):
        """Not enough time since last dream."""
        last_dream = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        trigger.write_last_dream_time(last_dream)
        assert trigger.check_time_gate() is False


class TestSessionGate:
    def test_passes_with_enough_sessions(self, trigger):
        """Enough sessions have accumulated."""
        assert trigger.check_session_gate(session_count=5) is True
        assert trigger.check_session_gate(session_count=10) is True

    def test_fails_with_too_few(self, trigger):
        """Not enough sessions."""
        assert trigger.check_session_gate(session_count=0) is False
        assert trigger.check_session_gate(session_count=4) is False


class TestLockGate:
    def test_passes_when_no_lock(self, trigger):
        """No other dream is running."""
        acquired = trigger.try_acquire_lock()
        assert acquired is True
        trigger.release_lock()

    def test_fails_when_locked(self, trigger):
        """Another dream holds the lock."""
        acquired1 = trigger.try_acquire_lock()
        assert acquired1 is True

        # Second attempt should fail (non-blocking)
        trigger2 = DreamTrigger(
            memory_dir=trigger._memory_dir,
            interval_hours=24,
            min_sessions=5,
        )
        acquired2 = trigger2.try_acquire_lock()
        assert acquired2 is False

        trigger.release_lock()

    def test_release_allows_reacquire(self, trigger):
        """After release, lock can be acquired again."""
        trigger.try_acquire_lock()
        trigger.release_lock()
        assert trigger.try_acquire_lock() is True
        trigger.release_lock()


class TestShouldDream:
    def test_all_gates_pass(self, trigger):
        """When all gates pass, should_dream returns True."""
        # No previous dream (time gate passes)
        # Enough sessions
        result = trigger.should_dream(session_count=5)
        assert result is True
        trigger.release_lock()

    def test_time_gate_blocks(self, trigger):
        """Recent dream blocks dreaming."""
        last_dream = datetime.now(tz=timezone.utc) - timedelta(hours=1)
        trigger.write_last_dream_time(last_dream)
        result = trigger.should_dream(session_count=10)
        assert result is False

    def test_session_gate_blocks(self, trigger):
        """Too few sessions blocks dreaming."""
        result = trigger.should_dream(session_count=2)
        assert result is False

    def test_lock_gate_blocks(self, trigger):
        """Held lock blocks dreaming."""
        trigger.try_acquire_lock()
        trigger2 = DreamTrigger(
            memory_dir=trigger._memory_dir,
            interval_hours=24,
            min_sessions=5,
        )
        result = trigger2.should_dream(session_count=10)
        assert result is False
        trigger.release_lock()
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_dream/test_trigger.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 4: Add filelock dependency**

Update `pyproject.toml` to add `"filelock>=3.15"` to dependencies, then run:

```bash
uv pip install -e ".[dev]"
```

**Step 5: Write implementation**

```python
# src/norn/dream/trigger.py
"""Dream trigger: three-gate system for deciding when to consolidate memory."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
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
        threshold = datetime.now(tz=timezone.utc) - timedelta(hours=self._interval_hours)
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
        if not self.try_acquire_lock():
            return False
        return True
```

**Step 6: Run tests to verify they pass**

Run: `uv run pytest tests/test_dream/test_trigger.py -v`
Expected: 11 PASSED

**Step 7: Run ruff**

Run: `uv run ruff check src/norn/dream/trigger.py`

**Step 8: Commit**

```bash
git add src/norn/dream/ tests/test_dream/ pyproject.toml
git commit -m "feat(dream): add three-gate dream trigger with time, session, and lock gates"
```

---

## Task 4: Dream Prompts -- LLM Instructions for Consolidation

**Files:**
- Create: `src/norn/dream/prompts.py`
- Create: `tests/test_dream/test_prompts.py`

The dream prompts define the system prompt and user prompt templates for each phase of consolidation. These are plain string templates, no LLM call yet.

**Step 1: Write the failing tests**

```python
# tests/test_dream/test_prompts.py
"""Tests for dream prompt templates."""

from norn.dream.prompts import build_consolidation_prompt, build_dream_system_prompt


def test_dream_system_prompt_is_string():
    prompt = build_dream_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    assert "memory" in prompt.lower()


def test_dream_system_prompt_has_rules():
    prompt = build_dream_system_prompt()
    assert "MEMORY.md" in prompt
    assert "200 lines" in prompt or "200" in prompt
    assert "absolute dates" in prompt.lower() or "relative dates" in prompt.lower()


def test_consolidation_prompt_includes_memory():
    prompt = build_consolidation_prompt(
        current_memory="# Norn Memory\n\n- User prefers Python",
        daily_logs={"2026-03-31": "Session 1: refactored auth"},
        topic_files={"auth": "# Auth\n\nUsing JWT."},
    )
    assert "User prefers Python" in prompt
    assert "refactored auth" in prompt
    assert "JWT" in prompt


def test_consolidation_prompt_handles_empty():
    prompt = build_consolidation_prompt(
        current_memory="",
        daily_logs={},
        topic_files={},
    )
    assert isinstance(prompt, str)
    assert "no existing memory" in prompt.lower() or "empty" in prompt.lower()


def test_consolidation_prompt_multiple_days():
    prompt = build_consolidation_prompt(
        current_memory="# Norn Memory",
        daily_logs={
            "2026-03-31": "Session 1: did X",
            "2026-03-30": "Session 2: did Y",
        },
        topic_files={},
    )
    assert "2026-03-31" in prompt
    assert "2026-03-30" in prompt
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dream/test_prompts.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/dream/prompts.py
"""Dream system prompts for memory consolidation."""

from __future__ import annotations


def build_dream_system_prompt() -> str:
    """Build the system prompt for the dream consolidation LLM call."""
    return """\
You are Norn's Dream Engine -- a memory consolidation system.

Your job is to process recent session logs and update the persistent memory files.
You are maintaining a knowledge base about the user, their projects, preferences, and ongoing work.

## Rules

1. **MEMORY.md** is the main index. Keep it under 200 lines and ~25KB.
2. Use **absolute dates** (2026-03-31), never relative dates (yesterday, last week).
3. **Delete contradicted facts** -- if new info contradicts old info, keep only the new.
4. **Merge duplicates** -- don't repeat the same fact in multiple places.
5. **Be concise** -- bullet points, not paragraphs.
6. **Preserve structure** -- maintain headers and sections in MEMORY.md.
7. **Topic files** -- create/update topic files for major ongoing projects or themes.
8. **Prune stale info** -- remove facts that are clearly outdated (> 30 days with no references).

## Output Format

Return a JSON object with these fields:
- `memory`: string -- the updated MEMORY.md content
- `topics`: dict[str, str] -- topic name -> content (only include topics that changed)
- `pruned_topics`: list[str] -- topic names to delete (stale/empty)
- `summary`: string -- one-line description of what changed
"""


def build_consolidation_prompt(
    current_memory: str,
    daily_logs: dict[str, str],
    topic_files: dict[str, str],
) -> str:
    """Build the user prompt with all context for consolidation."""
    parts: list[str] = []

    # Current memory
    if current_memory.strip():
        parts.append("## Current MEMORY.md\n")
        parts.append(f"```markdown\n{current_memory}\n```\n")
    else:
        parts.append("## Current MEMORY.md\n")
        parts.append("(Empty -- no existing memory yet)\n")

    # Daily logs (most recent first)
    parts.append("## Recent Daily Logs\n")
    if daily_logs:
        for date in sorted(daily_logs.keys(), reverse=True):
            parts.append(f"### {date}\n")
            parts.append(f"{daily_logs[date]}\n")
    else:
        parts.append("(No daily logs to process)\n")

    # Topic files
    parts.append("## Existing Topic Files\n")
    if topic_files:
        for name, content in sorted(topic_files.items()):
            parts.append(f"### topics/{name}.md\n")
            parts.append(f"```markdown\n{content}\n```\n")
    else:
        parts.append("(No topic files yet)\n")

    parts.append(
        "\nPlease consolidate the above into updated memory. "
        "Return JSON as specified in the system prompt."
    )

    return "\n".join(parts)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dream/test_prompts.py -v`
Expected: 5 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/dream/prompts.py`

**Step 6: Commit**

```bash
git add src/norn/dream/prompts.py tests/test_dream/test_prompts.py
git commit -m "feat(dream): add prompt templates for memory consolidation"
```

---

## Task 5: DreamEngine -- Four-Phase Consolidation

**Files:**
- Create: `src/norn/dream/engine.py`
- Create: `tests/test_dream/test_engine.py`

The DreamEngine orchestrates the 4-phase consolidation:
1. **Orient**: Read memory directory, MEMORY.md, existing topic files
2. **Gather**: Collect recent daily logs (new signal since last dream)
3. **Consolidate**: Send context to LLM, get updated memory
4. **Prune**: Apply updates, delete stale files, update last-dream timestamp

The engine uses an LLM provider (same protocol as the agent) for the consolidation call. The LLM returns structured JSON with updated memory content.

**Step 1: Write the failing tests**

```python
# tests/test_dream/test_engine.py
"""Tests for the dream engine."""

import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from norn.core.models import LLMResponse
from norn.dream.engine import DreamEngine, DreamResult
from norn.memory.models import MemoryConfig
from norn.memory.store import MemoryStore


@pytest.fixture
def memory_dir(tmp_path):
    return tmp_path / "memory"


@pytest.fixture
def store(memory_dir):
    config = MemoryConfig(memory_dir=memory_dir)
    s = MemoryStore(config)
    s.ensure_dirs()
    return s


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid consolidation JSON."""
    llm = AsyncMock()
    response_json = {
        "memory": "# Norn Memory\n\n- User works on project Norn\n- Prefers Python + uv\n",
        "topics": {"norn": "# Norn Project\n\nA coding agent built from scratch.\n"},
        "pruned_topics": [],
        "summary": "Added Norn project info from recent sessions.",
    }
    llm.complete = AsyncMock(
        return_value=LLMResponse(content=json.dumps(response_json))
    )
    return llm


@pytest.fixture
def engine(store, mock_llm):
    return DreamEngine(store=store, llm=mock_llm)


class TestOrientPhase:
    def test_reads_current_memory(self, engine, store):
        """Orient phase reads MEMORY.md."""
        context = engine._orient()
        assert "Norn Memory" in context["current_memory"]

    def test_reads_topic_files(self, engine, store):
        """Orient phase reads all topic files."""
        store.write_topic("auth", "# Auth\n\nJWT tokens.\n")
        context = engine._orient()
        assert "auth" in context["topic_files"]
        assert "JWT" in context["topic_files"]["auth"]

    def test_reads_empty_state(self, engine):
        """Orient phase handles empty memory."""
        context = engine._orient()
        assert context["topic_files"] == {}


class TestGatherPhase:
    def test_gathers_daily_logs(self, engine, store):
        """Gather phase collects daily logs."""
        store.append_daily("2026-03-31", "Session 1: did X")
        store.append_daily("2026-03-30", "Session 2: did Y")
        logs = engine._gather()
        assert "2026-03-31" in logs
        assert "2026-03-30" in logs

    def test_gathers_empty_logs(self, engine):
        """Gather phase handles no logs."""
        logs = engine._gather()
        assert logs == {}


class TestConsolidatePhase:
    @pytest.mark.asyncio
    async def test_calls_llm(self, engine, mock_llm, store):
        """Consolidate phase calls the LLM with context."""
        store.append_daily("2026-03-31", "Session 1: built agent")
        result = await engine._consolidate(
            current_memory="# Norn Memory\n",
            daily_logs={"2026-03-31": "Session 1: built agent"},
            topic_files={},
        )
        mock_llm.complete.assert_called_once()
        assert result is not None
        assert "memory" in result

    @pytest.mark.asyncio
    async def test_parses_json_response(self, engine, store):
        """Consolidate phase parses the LLM JSON response."""
        result = await engine._consolidate(
            current_memory="",
            daily_logs={},
            topic_files={},
        )
        assert "Norn Memory" in result["memory"]
        assert "norn" in result["topics"]


class TestPrunePhase:
    def test_applies_memory_update(self, engine, store):
        """Prune phase writes updated MEMORY.md."""
        engine._prune(
            consolidation={
                "memory": "# Updated Memory\n\n- new fact\n",
                "topics": {},
                "pruned_topics": [],
                "summary": "test",
            }
        )
        content = store.read_memory()
        assert "Updated Memory" in content
        assert "new fact" in content

    def test_applies_topic_updates(self, engine, store):
        """Prune phase writes updated topic files."""
        engine._prune(
            consolidation={
                "memory": "# Memory\n",
                "topics": {"auth": "# Auth\n\nOAuth2 now.\n"},
                "pruned_topics": [],
                "summary": "test",
            }
        )
        content = store.read_topic("auth")
        assert "OAuth2" in content

    def test_prunes_stale_topics(self, engine, store):
        """Prune phase deletes stale topic files."""
        store.write_topic("old-project", "# Old\n\nNo longer relevant.\n")
        assert store.read_topic("old-project") is not None

        engine._prune(
            consolidation={
                "memory": "# Memory\n",
                "topics": {},
                "pruned_topics": ["old-project"],
                "summary": "test",
            }
        )
        assert store.read_topic("old-project") is None


class TestFullDream:
    @pytest.mark.asyncio
    async def test_dream_end_to_end(self, engine, store):
        """Full dream cycle: orient -> gather -> consolidate -> prune."""
        store.append_daily("2026-03-31", "Session 1: worked on agent")
        result = await engine.dream()
        assert isinstance(result, DreamResult)
        assert result.success is True
        assert result.summary is not None
        # Memory should be updated
        content = store.read_memory()
        assert "Norn" in content

    @pytest.mark.asyncio
    async def test_dream_handles_llm_error(self, store):
        """Dream handles LLM errors gracefully."""
        bad_llm = AsyncMock()
        bad_llm.complete = AsyncMock(side_effect=Exception("API error"))
        engine = DreamEngine(store=store, llm=bad_llm)
        result = await engine.dream()
        assert result.success is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_dream_handles_invalid_json(self, store):
        """Dream handles invalid JSON from LLM."""
        bad_llm = AsyncMock()
        bad_llm.complete = AsyncMock(
            return_value=LLMResponse(content="this is not json")
        )
        engine = DreamEngine(store=store, llm=bad_llm)
        result = await engine.dream()
        assert result.success is False
        assert result.error is not None
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_dream/test_engine.py -v`
Expected: FAIL -- `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/dream/engine.py
"""Dream engine: four-phase memory consolidation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from norn.core.models import Message, Role
from norn.dream.prompts import build_consolidation_prompt, build_dream_system_prompt

if TYPE_CHECKING:
    from norn.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream consolidation."""

    success: bool
    summary: str | None = None
    error: str | None = None
    files_written: list[str] = field(default_factory=list)
    files_pruned: list[str] = field(default_factory=list)


class DreamEngine:
    """Orchestrates the four-phase memory consolidation."""

    def __init__(self, store: MemoryStore, llm: object) -> None:
        self._store = store
        self._llm = llm

    # --- Phase 1: Orient ---

    def _orient(self) -> dict[str, Any]:
        """Read current memory state."""
        return {
            "current_memory": self._store.read_memory(),
            "topic_files": {
                name: self._store.read_topic(name) or ""
                for name in self._store.list_topics()
            },
        }

    # --- Phase 2: Gather ---

    def _gather(self) -> dict[str, str]:
        """Collect recent daily logs."""
        logs: dict[str, str] = {}
        for date_str in self._store.list_daily_logs():
            content = self._store.read_daily(date_str)
            if content:
                logs[date_str] = content
        return logs

    # --- Phase 3: Consolidate ---

    async def _consolidate(
        self,
        current_memory: str,
        daily_logs: dict[str, str],
        topic_files: dict[str, str],
    ) -> dict[str, Any]:
        """Send context to LLM and get consolidated memory."""
        system_prompt = build_dream_system_prompt()
        user_prompt = build_consolidation_prompt(
            current_memory=current_memory,
            daily_logs=daily_logs,
            topic_files=topic_files,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]

        response = await self._llm.complete(messages=messages, tools=None)
        content = response.content or ""

        # Parse JSON from response (handle markdown code blocks)
        json_str = content.strip()
        if json_str.startswith("```"):
            # Strip markdown code block markers
            lines = json_str.split("\n")
            json_str = "\n".join(lines[1:-1]) if len(lines) > 2 else json_str

        return json.loads(json_str)

    # --- Phase 4: Prune ---

    def _prune(self, consolidation: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Apply consolidation results to memory files."""
        files_written: list[str] = []
        files_pruned: list[str] = []

        # Update MEMORY.md
        memory_content = consolidation.get("memory", "")
        if memory_content:
            self._store.write_memory(memory_content)
            files_written.append("MEMORY.md")

        # Update topic files
        topics = consolidation.get("topics", {})
        for name, content in topics.items():
            self._store.write_topic(name, content)
            files_written.append(f"topics/{name}.md")

        # Delete pruned topics
        pruned = consolidation.get("pruned_topics", [])
        for name in pruned:
            topic_path = self._store._topics_dir / f"{name}.md"
            if topic_path.exists():
                topic_path.unlink()
                files_pruned.append(f"topics/{name}.md")

        return files_written, files_pruned

    # --- Full dream cycle ---

    async def dream(self) -> DreamResult:
        """Run the complete four-phase dream consolidation."""
        try:
            # Phase 1: Orient
            context = self._orient()

            # Phase 2: Gather
            daily_logs = self._gather()

            # Phase 3: Consolidate
            consolidation = await self._consolidate(
                current_memory=context["current_memory"],
                daily_logs=daily_logs,
                topic_files=context["topic_files"],
            )

            # Phase 4: Prune
            files_written, files_pruned = self._prune(consolidation)

            summary = consolidation.get("summary", "Memory consolidated.")
            return DreamResult(
                success=True,
                summary=summary,
                files_written=files_written,
                files_pruned=files_pruned,
            )

        except json.JSONDecodeError as e:
            logger.error("Dream failed: invalid JSON from LLM: %s", e)
            return DreamResult(success=False, error=f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error("Dream failed: %s", e)
            return DreamResult(success=False, error=str(e))
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_dream/test_engine.py -v`
Expected: 12 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/dream/engine.py`

**Step 6: Commit**

```bash
git add src/norn/dream/engine.py tests/test_dream/test_engine.py
git commit -m "feat(dream): add four-phase dream engine with LLM-based memory consolidation"
```

---

## Task 6: Memory Config Integration

**Files:**
- Modify: `src/norn/core/config.py`
- Modify: `tests/test_core/test_config.py`

Add memory/dream configuration to `NornConfig` so the user can customize memory directory, dream intervals, and dream model.

**Step 1: Write the failing tests**

Add to `tests/test_core/test_config.py`:

```python
def test_memory_config_defaults():
    """NornConfig should have memory config with defaults."""
    config = NornConfig()
    assert config.memory.enabled is True
    assert config.memory.dream_interval_hours == 24
    assert config.memory.dream_min_sessions == 5


def test_memory_config_custom():
    """Memory config should be overridable."""
    config = NornConfig(
        memory={"dream_interval_hours": 12, "dream_min_sessions": 3}
    )
    assert config.memory.dream_interval_hours == 12
    assert config.memory.dream_min_sessions == 3
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: FAIL -- `NornConfig` has no field `memory`

**Step 3: Modify config.py**

Add a `MemorySystemConfig` model and include it in `NornConfig`:

```python
class MemorySystemConfig(BaseModel):
    enabled: bool = True
    memory_dir: str = "~/.norn/memory"
    dream_interval_hours: int = 24
    dream_min_sessions: int = 5
    dream_model: str | None = None  # None = use main LLM
```

Add to `NornConfig`:

```python
class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: ALL PASSED

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED (no regressions)

**Step 6: Commit**

```bash
git add src/norn/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): add memory system configuration with dream intervals and model"
```

---

## Task 7: AgentLoop Integration -- Memory Injection + Session Tracking

**Files:**
- Modify: `src/norn/core/agent.py`
- Modify: `tests/test_core/test_agent.py`

The AgentLoop now optionally accepts a `MemoryStore` and `SessionLogger`. If present:
- MEMORY.md content is prepended to the system prompt
- Session metadata is tracked (message count, tool call count)
- At session end, a `SessionSummary` is logged

**Step 1: Write the failing tests**

Add to `tests/test_core/test_agent.py`:

```python
from norn.memory.models import MemoryConfig
from norn.memory.store import MemoryStore
from norn.memory.session_logger import SessionLogger


@pytest.fixture
def memory_store(tmp_path):
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.write_memory("# Norn Memory\n\n- User likes Python\n- Project: Norn agent\n")
    return store


@pytest.fixture
def session_logger(memory_store):
    return SessionLogger(memory_store)


@pytest.mark.asyncio
async def test_agent_injects_memory_into_system_prompt(registry, memory_store):
    """When memory_store is provided, MEMORY.md is added to system prompt."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="I know you like Python!", tool_calls=[])
    )

    agent = AgentLoop(llm=llm, registry=registry, memory_store=memory_store)
    await agent.run("What do you know about me?")

    # Check that the system prompt sent to LLM includes memory
    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    system_msg = messages[0]
    assert "User likes Python" in system_msg.content


@pytest.mark.asyncio
async def test_agent_tracks_session_stats(registry, memory_store, session_logger):
    """Agent tracks message and tool call counts for session logging."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="Done.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(
        llm=llm, registry=registry,
        memory_store=memory_store, session_logger=session_logger,
    )
    await agent.run("echo hi")
    assert agent.tool_call_count == 1
    assert agent.user_message_count == 1


@pytest.mark.asyncio
async def test_agent_without_memory_still_works(registry):
    """Backward compatibility: agent works without memory."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="hello", tool_calls=[])
    )
    agent = AgentLoop(llm=llm, registry=registry)
    result = await agent.run("hi")
    assert result.content == "hello"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_agent.py -v`
Expected: FAIL -- `AgentLoop() got an unexpected keyword argument 'memory_store'`

**Step 3: Modify AgentLoop**

Add optional `memory_store` and `session_logger` parameters. Inject memory into system prompt. Track stats.

Key changes to `src/norn/core/agent.py`:

```python
if TYPE_CHECKING:
    from norn.memory.session_logger import SessionLogger
    from norn.memory.store import MemoryStore
    from norn.permissions.checker import PermissionChecker
    from norn.tools.registry import ToolRegistry


class AgentLoop:
    MAX_TOOL_ROUNDS = 25

    def __init__(
        self,
        llm: object,
        registry: ToolRegistry,
        system_prompt: str = "You are Norn, a helpful coding agent.",
        cwd: str = ".",
        permission_checker: PermissionChecker | None = None,
        memory_store: MemoryStore | None = None,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cwd=cwd)
        self.history: list[Message] = []
        self.permission_checker = permission_checker
        self.memory_store = memory_store
        self.session_logger = session_logger
        # Session stats
        self.user_message_count = 0
        self.tool_call_count = 0

    def _build_system_prompt(self) -> str:
        """Build system prompt with optional memory injection."""
        prompt = self.system_prompt
        if self.memory_store is not None:
            memory_content = self.memory_store.read_memory()
            if memory_content.strip():
                prompt += "\n\n## Persistent Memory\n\n" + memory_content
        return prompt

    async def run(self, user_input: str) -> LLMResponse:
        self.user_message_count += 1
        self.history.append(Message(role=Role.USER, content=user_input))

        messages = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            *self.history,
        ]

        for _round in range(self.MAX_TOOL_ROUNDS):
            response = await self.llm.complete(
                messages=messages,
                tools=self.registry.get_schemas() or None,
            )

            if not response.has_tool_calls:
                self.history.append(Message(role=Role.ASSISTANT, content=response.content))
                return response

            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)

            for call in response.tool_calls:
                self.tool_call_count += 1
                result = await self._execute_tool(call)
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output or result.error or "",
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)

        final = LLMResponse(content="[Max tool rounds reached]")
        self.history.append(Message(role=Role.ASSISTANT, content=final.content))
        return final
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_agent.py -v`
Expected: ALL PASSED (old + new)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 6: Commit**

```bash
git add src/norn/core/agent.py tests/test_core/test_agent.py
git commit -m "feat(core): integrate memory store and session tracking into agent loop"
```

---

## Task 8: CLI Wiring -- Memory + Dream in Commands

**Files:**
- Modify: `src/norn/cli/main.py`

Wire the memory store, session logger, and dream trigger into the CLI commands. Add a `norn dream` command for manual dream triggering. Update `chat`/`run` to use memory.

**Step 1: Update CLI**

Key additions to `src/norn/cli/main.py`:

```python
from norn.memory.models import MemoryConfig
from norn.memory.store import MemoryStore
from norn.memory.session_logger import SessionLogger


def _build_memory_store(config: NornConfig) -> MemoryStore | None:
    """Build memory store if memory is enabled."""
    if not config.memory.enabled:
        return None
    mem_config = MemoryConfig(
        memory_dir=Path(config.memory.memory_dir).expanduser(),
        dream_interval_hours=config.memory.dream_interval_hours,
        dream_min_sessions=config.memory.dream_min_sessions,
    )
    store = MemoryStore(mem_config)
    store.ensure_dirs()
    return store
```

Update `chat()` and `run()` to pass `memory_store` and `session_logger` to `AgentLoop`.

Add new command:

```python
@app.command()
def dream() -> None:
    """Manually trigger a memory consolidation dream."""
    config = NornConfig.load()
    config.apply_env_overrides()

    store = _build_memory_store(config)
    if store is None:
        console.print("[yellow]Memory system is disabled.[/yellow]")
        raise typer.Exit(1)

    provider = _build_provider(config)

    from norn.dream.engine import DreamEngine
    engine = DreamEngine(store=store, llm=provider)

    async def _dream() -> None:
        with console.status("[dim]Dreaming...[/dim]"):
            result = await engine.dream()
        if result.success:
            console.print(f"[green]Dream complete:[/green] {result.summary}")
            if result.files_written:
                console.print(f"  Written: {', '.join(result.files_written)}")
            if result.files_pruned:
                console.print(f"  Pruned: {', '.join(result.files_pruned)}")
        else:
            console.print(f"[red]Dream failed:[/red] {result.error}")

    asyncio.run(_dream())
```

Update `config` command to display memory status.

**Step 2: Verify CLI**

Run: `uv run norn --help` (should show `dream` command)
Run: `uv run norn config` (should show memory status)

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED (no regressions)

**Step 4: Commit**

```bash
git add src/norn/cli/main.py
git commit -m "feat(cli): wire memory store, session logger, and dream command into CLI"
```

---

## Task 9: Integration Tests

**Files:**
- Modify: `tests/test_integration.py`

Add integration tests that exercise the full memory + dream pipeline.

**Step 1: Write integration tests**

Add to `tests/test_integration.py`:

```python
import json
from norn.memory.models import MemoryConfig, SessionSummary
from norn.memory.store import MemoryStore
from norn.memory.session_logger import SessionLogger
from norn.dream.engine import DreamEngine
from norn.dream.trigger import DreamTrigger
from datetime import datetime, timezone


@pytest.mark.asyncio
async def test_agent_with_memory_injects_context(tmp_path):
    """Full pipeline: memory content appears in system prompt sent to LLM."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.write_memory("# Memory\n\n- User's project is called Norn\n")

    registry = ToolRegistry()
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="Your project is Norn!", tool_calls=[])
    )

    agent = AgentLoop(llm=llm, registry=registry, memory_store=store)
    await agent.run("What's my project?")

    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    system_content = messages[0].content
    assert "Norn" in system_content


@pytest.mark.asyncio
async def test_dream_engine_full_cycle(tmp_path):
    """Full dream cycle: orient -> gather -> consolidate -> prune."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.append_daily("2026-03-31", "Session 1: built the dream system")

    response_json = {
        "memory": "# Norn Memory\n\n- Built dream system on 2026-03-31\n",
        "topics": {},
        "pruned_topics": [],
        "summary": "Recorded dream system work.",
    }
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content=json.dumps(response_json))
    )

    engine = DreamEngine(store=store, llm=llm)
    result = await engine.dream()
    assert result.success is True
    assert "dream system" in store.read_memory().lower()


def test_dream_trigger_gates(tmp_path):
    """Integration: trigger respects all three gates."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    trigger = DreamTrigger(
        memory_dir=memory_dir,
        interval_hours=24,
        min_sessions=5,
    )

    # Should dream: no previous dream + enough sessions
    assert trigger.should_dream(session_count=5) is True
    trigger.release_lock()

    # Record dream time as now
    trigger.write_last_dream_time(datetime.now(tz=timezone.utc))

    # Should NOT dream: too recent
    assert trigger.should_dream(session_count=10) is False


@pytest.mark.asyncio
async def test_session_logger_records_and_counts(tmp_path):
    """Session logger records summaries and tracks count."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    logger = SessionLogger(store)

    for i in range(3):
        summary = SessionSummary(
            session_id=f"s{i}",
            started_at=datetime(2026, 3, 31, 10 + i, 0, tzinfo=timezone.utc),
            ended_at=datetime(2026, 3, 31, 10 + i, 30, tzinfo=timezone.utc),
            user_messages=3,
            tool_calls=5,
            summary=f"Session {i} summary.",
        )
        logger.log_session(summary)

    assert logger.session_count() == 3
    daily_content = store.read_daily("2026-03-31")
    assert "s0" in daily_content
    assert "s1" in daily_content
    assert "s2" in daily_content
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: ALL PASSED

**Step 3: Run full test suite + ruff**

Run: `uv run pytest -v && uv run ruff check src/ tests/`
Expected: ALL PASSED

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for memory system and dream engine"
```

---

## Task 10: Final Verification and Cleanup

**Step 1: Run full test suite**

```bash
uv run pytest -v
```
Expected: ALL PASSED

**Step 2: Run ruff**

```bash
uv run ruff check src/ tests/
```
Expected: 0 errors (except pre-existing E402 in CLI)

**Step 3: Verify git log**

```bash
git log --oneline
```
Expected: Clean atomic commits from Phase 3

**Step 4: Manual smoke test**

```bash
# Create a memory and test dream
uv run norn config
```
Expected: Shows memory system as enabled

**Step 5: Final commit (if any cleanup needed)**

```bash
# Only if cleanup was needed
git add -A && git commit -m "chore: phase 3 cleanup"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 0 | Memory models | 6 | `memory/models.py` |
| 1 | Memory store | 16 | `memory/store.py` |
| 2 | Session logger | 6 | `memory/session_logger.py` |
| 3 | Dream trigger (3-gate) | 11 | `dream/trigger.py` |
| 4 | Dream prompts | 5 | `dream/prompts.py` |
| 5 | Dream engine (4-phase) | 12 | `dream/engine.py` |
| 6 | Memory config | 2 | `core/config.py` (modify) |
| 7 | AgentLoop integration | 3 | `core/agent.py` (modify) |
| 8 | CLI wiring | manual | `cli/main.py` (modify) |
| 9 | Integration tests | 4 | `test_integration.py` (modify) |
| 10 | Final verification | manual | -- |

**Total: ~65 new tests, 10 commits, 5 new files, 3 modified files**

**New dependency**: `filelock>=3.15`
