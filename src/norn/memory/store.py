"""Memory store: manages reading and writing memory files."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.memory.models import MemoryConfig

_MEMORY_HEADER = """# Norn Memory

> Auto-maintained by the dream system. Do not edit manually unless you know what you're doing.

"""


def _safe_name(name: str) -> str:
    """Sanitize a name to prevent path traversal."""
    sanitized = Path(name).name
    if not sanitized or sanitized.startswith("."):
        msg = f"Invalid name: {name!r}"
        raise ValueError(msg)
    return sanitized


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
        path = self._topics_dir / f"{_safe_name(name)}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def write_topic(self, name: str, content: str) -> None:
        """Write (overwrite) a topic file."""
        path = self._topics_dir / f"{_safe_name(name)}.md"
        path.write_text(content, encoding="utf-8")

    def list_topics(self) -> list[str]:
        """List all topic names (without .md extension)."""
        if not self._topics_dir.exists():
            return []
        return sorted(p.stem for p in self._topics_dir.glob("*.md"))

    def delete_topic(self, name: str) -> bool:
        """Delete a topic file. Returns True if the file existed."""
        path = self._topics_dir / f"{_safe_name(name)}.md"
        if path.exists():
            path.unlink()
            return True
        return False

    # --- Lessons (W3.1 self-improvement) ---

    @property
    def _lessons_file(self) -> Path:
        return self._topics_dir / "lessons.md"

    def read_lessons(self) -> str:
        """Return the learned lessons file content, or empty string."""
        if not self._lessons_file.exists():
            return ""
        return self._lessons_file.read_text(encoding="utf-8")

    def append_lesson(self, lesson: str) -> None:
        """Append a lesson entry. Creates the file on first call."""
        with open(self._lessons_file, "a", encoding="utf-8") as f:
            f.write(lesson.rstrip() + "\n\n")

    def backup_lessons(self) -> Path | None:
        """Copy lessons.md → lessons.md.bak (pre-change safety net, W3.3).

        Returns the backup path, or None when there is no lessons file yet
        (in which case the backup is an empty file, so a later restore
        reverts to the pre-first-lesson state).
        """
        bak = self._lessons_file.with_suffix(".md.bak")
        if self._lessons_file.exists():
            bak.write_text(self._lessons_file.read_text(encoding="utf-8"), encoding="utf-8")
            return bak
        bak.write_text("", encoding="utf-8")
        return None

    def restore_lessons_backup(self) -> bool:
        """Restore lessons.md from lessons.md.bak. Returns True if restored."""
        bak = self._lessons_file.with_suffix(".md.bak")
        if not bak.exists():
            return False
        content = bak.read_text(encoding="utf-8")
        if content:
            self._lessons_file.write_text(content, encoding="utf-8")
        elif self._lessons_file.exists():
            self._lessons_file.unlink()
        return True

    # --- Daily logs ---

    def append_daily(self, date_str: str, entry: str) -> None:
        """Append an entry to a daily log file."""
        path = self._daily_dir / f"{_safe_name(date_str)}.md"
        with open(path, "a", encoding="utf-8") as f:
            f.write(entry + "\n\n")

    def read_daily(self, date_str: str) -> str | None:
        """Read a daily log. Returns None if it doesn't exist."""
        path = self._daily_dir / f"{_safe_name(date_str)}.md"
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_daily_logs(self) -> list[str]:
        """List daily log dates, most recent first."""
        if not self._daily_dir.exists():
            return []
        dates = sorted((p.stem for p in self._daily_dir.glob("*.md")), reverse=True)
        return dates
