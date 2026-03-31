"""Tests for the coordinator scratchpad."""

from __future__ import annotations

import pytest

from norn.coordinator.scratchpad import Scratchpad


@pytest.fixture
def scratchpad(tmp_path):
    """Create a scratchpad in a temp directory."""
    return Scratchpad(base_dir=tmp_path / "scratchpad")


class TestInit:
    def test_creates_directory_structure(self, scratchpad):
        """Scratchpad creates phase directories on init."""
        scratchpad.ensure_dirs()
        assert (scratchpad.base_dir / "research").is_dir()
        assert (scratchpad.base_dir / "synthesis").is_dir()
        assert (scratchpad.base_dir / "implementation").is_dir()
        assert (scratchpad.base_dir / "verification").is_dir()

    def test_creates_base_dir(self, scratchpad):
        """Base directory is created if it doesn't exist."""
        scratchpad.ensure_dirs()
        assert scratchpad.base_dir.is_dir()


class TestReadWrite:
    def test_write_and_read_section(self, scratchpad):
        """Write to a section and read it back."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "worker-1-findings.md", "# Findings\n\nFound 3 issues.")
        content = scratchpad.read("research", "worker-1-findings.md")
        assert "Found 3 issues" in content

    def test_read_nonexistent_returns_none(self, scratchpad):
        """Reading a file that doesn't exist returns None."""
        scratchpad.ensure_dirs()
        assert scratchpad.read("research", "nonexistent.md") is None

    def test_write_multiple_files(self, scratchpad):
        """Multiple workers can write to the same section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "worker-a.md", "Findings A")
        scratchpad.write("research", "worker-b.md", "Findings B")
        assert scratchpad.read("research", "worker-a.md") == "Findings A"
        assert scratchpad.read("research", "worker-b.md") == "Findings B"

    def test_overwrite_existing(self, scratchpad):
        """Writing to an existing file overwrites it."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "notes.md", "v1")
        scratchpad.write("research", "notes.md", "v2")
        assert scratchpad.read("research", "notes.md") == "v2"


class TestListAndCollect:
    def test_list_section_files(self, scratchpad):
        """List all files in a section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("research", "b.md", "y")
        files = scratchpad.list_files("research")
        assert set(files) == {"a.md", "b.md"}

    def test_list_empty_section(self, scratchpad):
        """Empty section returns empty list."""
        scratchpad.ensure_dirs()
        assert scratchpad.list_files("research") == []

    def test_collect_section(self, scratchpad):
        """Collect all content from a section into a dict."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "w1.md", "Result 1")
        scratchpad.write("research", "w2.md", "Result 2")
        collected = scratchpad.collect("research")
        assert collected == {"w1.md": "Result 1", "w2.md": "Result 2"}

    def test_collect_empty_section(self, scratchpad):
        """Collect from empty section returns empty dict."""
        scratchpad.ensure_dirs()
        assert scratchpad.collect("research") == {}

    def test_collect_all(self, scratchpad):
        """Collect everything across all sections."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "r1.md", "Research 1")
        scratchpad.write("synthesis", "spec.md", "Spec content")
        all_content = scratchpad.collect_all()
        assert "research" in all_content
        assert "synthesis" in all_content
        assert all_content["research"]["r1.md"] == "Research 1"
        assert all_content["synthesis"]["spec.md"] == "Spec content"


class TestCleanup:
    def test_clear_section(self, scratchpad):
        """Clear removes all files from a section."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("research", "b.md", "y")
        scratchpad.clear("research")
        assert scratchpad.list_files("research") == []

    def test_clear_all(self, scratchpad):
        """Clear all sections."""
        scratchpad.ensure_dirs()
        scratchpad.write("research", "a.md", "x")
        scratchpad.write("synthesis", "b.md", "y")
        scratchpad.clear_all()
        assert scratchpad.list_files("research") == []
        assert scratchpad.list_files("synthesis") == []
