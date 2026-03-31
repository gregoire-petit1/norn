"""Scratchpad: filesystem-based communication between coordinator and workers."""

from __future__ import annotations

from pathlib import Path

from norn.coordinator.models import CoordinatorPhase


def _safe_name(name: str) -> str:
    """Sanitize a filename to prevent path traversal."""
    sanitized = Path(name).name
    if not sanitized or sanitized.startswith("."):
        msg = f"Invalid filename: {name!r}"
        raise ValueError(msg)
    return sanitized


_SECTIONS = [phase.value for phase in CoordinatorPhase]


class Scratchpad:
    """Filesystem-backed scratchpad for inter-worker communication."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)

    def ensure_dirs(self) -> None:
        """Create the scratchpad directory structure."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        for section in _SECTIONS:
            (self.base_dir / section).mkdir(exist_ok=True)

    def write(self, section: str, filename: str, content: str) -> None:
        """Write a file to a scratchpad section."""
        path = self.base_dir / _safe_name(section) / _safe_name(filename)
        path.write_text(content, encoding="utf-8")

    def read(self, section: str, filename: str) -> str | None:
        """Read a file from a scratchpad section. Returns None if missing."""
        path = self.base_dir / _safe_name(section) / _safe_name(filename)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def list_files(self, section: str) -> list[str]:
        """List all files in a section."""
        section_dir = self.base_dir / section
        if not section_dir.exists():
            return []
        return sorted(p.name for p in section_dir.iterdir() if p.is_file())

    def collect(self, section: str) -> dict[str, str]:
        """Collect all file contents from a section."""
        result: dict[str, str] = {}
        for filename in self.list_files(section):
            content = self.read(section, filename)
            if content is not None:
                result[filename] = content
        return result

    def collect_all(self) -> dict[str, dict[str, str]]:
        """Collect all content across all sections."""
        return {section: self.collect(section) for section in _SECTIONS if self.list_files(section)}

    def clear(self, section: str) -> None:
        """Remove all files from a section."""
        section_dir = self.base_dir / section
        if section_dir.exists():
            for p in section_dir.iterdir():
                if p.is_file():
                    p.unlink()

    def clear_all(self) -> None:
        """Remove all files from all sections."""
        for section in _SECTIONS:
            self.clear(section)
