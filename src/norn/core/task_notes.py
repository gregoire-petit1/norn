"""Persistent task scratchpad for long-horizon runs (SOTA v2, wave 2 — A1).

A single task-scoped markdown file with fixed sections that the agent
maintains via a tool. Its rendered content is injected as a SYSTEM message
each turn (the same slot the sliding-window summary uses), so it survives
context compaction: even when old raw turns are evicted, the distilled
plan/progress is regenerated from the file and stays in front of the model.

Bounded by construction: sections are overwritten in place (not appended),
and render() hard-caps total size, evicting the oldest Progress lines first.
"""

from __future__ import annotations

from pathlib import Path

# Fixed section order. Plan/Decisions stay small (overwritten); Progress is
# the only append-shaped section and is the one render() trims under pressure.
_SECTIONS = ("Goal", "Plan", "Decisions", "Progress")

# Map the tool's action verbs to sections.
ACTION_SECTIONS = {
    "set_goal": "Goal",
    "set_plan": "Plan",
    "record_decision": "Decisions",
    "note_progress": "Progress",
}
# Sections whose entries accumulate (timestamped bullets) rather than replace.
_APPEND_SECTIONS = ("Decisions", "Progress")


class TaskNotes:
    """File-backed task scratchpad with bounded, section-structured content."""

    def __init__(self, path: Path, max_chars: int = 4000) -> None:
        self.path = Path(path)
        self.max_chars = max_chars

    def _load(self) -> dict[str, list[str]]:
        """Parse the file into {section: [lines]}. Missing file → empty."""
        sections: dict[str, list[str]] = {s: [] for s in _SECTIONS}
        if not self.path.exists():
            return sections
        current: str | None = None
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.startswith("## "):
                name = line[3:].strip()
                current = name if name in sections else None
                continue
            if current is not None and line.strip():
                sections[current].append(line.rstrip())
        return sections

    def _save(self, sections: dict[str, list[str]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        parts: list[str] = []
        for s in _SECTIONS:
            parts.append(f"## {s}")
            parts.extend(sections[s])
            parts.append("")
        self.path.write_text("\n".join(parts).rstrip() + "\n", encoding="utf-8")

    def write_section(self, section: str, content: str) -> None:
        """Set (overwrite) or append to a section, depending on its kind.

        Goal/Plan overwrite (single source of truth); Decisions/Progress
        append one bullet per call (the running log).
        """
        if section not in _SECTIONS:
            raise ValueError(f"Unknown section {section!r}; valid: {_SECTIONS}")
        sections = self._load()
        entry = content.strip()
        if not entry:
            return
        if section in _APPEND_SECTIONS:
            bullet = entry if entry.startswith("- ") else f"- {entry}"
            sections[section].append(bullet)
        else:
            sections[section] = entry.splitlines()
        self._save(sections)

    def read(self) -> str:
        """Return the raw file content, or empty string if none."""
        if not self.path.exists():
            return ""
        return self.path.read_text(encoding="utf-8")

    def render(self) -> str:
        """Render the notes for injection, clamped to max_chars.

        Under pressure, oldest Progress bullets are dropped first (they are
        the least load-bearing — Goal/Plan/Decisions orient the agent).
        Returns "" when there is no content, so the caller can skip injection.
        """
        sections = self._load()
        if not any(sections[s] for s in _SECTIONS):
            return ""

        def build(progress: list[str]) -> str:
            parts = ["# Task Notes (persists across context compaction)"]
            for s in _SECTIONS:
                lines = progress if s == "Progress" else sections[s]
                if lines:
                    parts.append(f"\n## {s}")
                    parts.extend(lines)
            return "\n".join(parts)

        progress = list(sections["Progress"])
        rendered = build(progress)
        # Evict oldest Progress bullets until it fits.
        while len(rendered) > self.max_chars and progress:
            progress.pop(0)
            rendered = build(progress)
        if len(rendered) > self.max_chars:
            rendered = rendered[: self.max_chars]
        return rendered
