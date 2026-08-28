"""Tests for the memory store."""

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


class TestLessons:
    def test_read_lessons_empty(self, store):
        """No lessons file → empty string."""
        store.ensure_dirs()
        assert store.read_lessons() == ""

    def test_append_and_read_lesson(self, store, memory_dir):
        """Appending a lesson creates the file and can be read back."""
        store.ensure_dirs()
        store.append_lesson("Always verify edge cases.")
        content = store.read_lessons()
        assert "Always verify edge cases." in content

    def test_append_multiple_lessons(self, store, memory_dir):
        """Multiple lessons are all present."""
        store.ensure_dirs()
        store.append_lesson("Lesson A.")
        store.append_lesson("Lesson B.")
        content = store.read_lessons()
        assert "Lesson A." in content
        assert "Lesson B." in content

    def test_lessons_stored_in_topics_dir(self, store, memory_dir):
        """lessons.md lives in the topics directory."""
        store.ensure_dirs()
        store.append_lesson("x")
        assert (memory_dir / "topics" / "lessons.md").exists()


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


class TestLessonsBackup:
    """W3.3 safety net: backup/restore around lesson persistence."""

    def test_backup_and_restore_round_trip(self, store):
        store.ensure_dirs()
        store.append_lesson("## Lesson 1\n\ngood lesson")
        store.backup_lessons()
        store.append_lesson("## Lesson 2\n\nbad lesson")
        assert "bad lesson" in store.read_lessons()

        assert store.restore_lessons_backup() is True
        content = store.read_lessons()
        assert "good lesson" in content
        assert "bad lesson" not in content

    def test_backup_before_first_lesson_restores_to_empty(self, store):
        store.ensure_dirs()
        store.backup_lessons()  # no lessons file yet
        store.append_lesson("## Lesson 1\n\nfirst")
        assert store.restore_lessons_backup() is True
        assert store.read_lessons() == ""

    def test_restore_without_backup_returns_false(self, store):
        store.ensure_dirs()
        assert store.restore_lessons_backup() is False
