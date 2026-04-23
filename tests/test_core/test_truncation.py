"""Tests for tool output truncation (W1.2)."""

from __future__ import annotations

import pytest

from norn.core.truncation import truncate_tool_output


class TestTruncateToolOutput:
    """Tests for the bookend truncation strategy."""

    def test_short_text_unchanged(self):
        """Text shorter than max_chars should be returned unchanged."""
        text = "Hello, world!"
        result = truncate_tool_output(text, max_chars=1000)
        assert result == text

    def test_exact_max_chars_unchanged(self):
        """Text exactly at max_chars should be returned unchanged."""
        text = "x" * 500
        result = truncate_tool_output(text, max_chars=500)
        assert result == text

    def test_bookend_includes_head_and_tail(self):
        """Truncated output should contain the beginning and end of the text."""
        # Create a text with clear beginning and end markers
        head = "HEAD_MARKER " * 100  # ~1200 chars
        middle = "MIDDLE " * 500  # ~3500 chars
        tail = "TAIL_MARKER " * 100  # ~1200 chars
        text = head + middle + tail

        result = truncate_tool_output(text, max_chars=3000)

        assert "HEAD_MARKER" in result
        assert "TAIL_MARKER" in result
        assert len(result) <= 3000 + 100  # some tolerance for the marker

    def test_bookend_contains_truncation_marker(self):
        """Truncated output should contain the '[truncated ...]' marker."""
        text = "x" * 5000
        result = truncate_tool_output(text, max_chars=2000)

        assert "...[truncated" in result
        assert "chars]..." in result

    def test_bookend_head_ratio(self):
        """Head portion should be ~60% of max_chars."""
        lines = [f"line-{i:04d}" for i in range(500)]
        text = "\n".join(lines)

        result = truncate_tool_output(text, max_chars=2000)

        # The head portion (before the marker) should be roughly 1200 chars (60%)
        marker_pos = result.index("...[truncated")
        head_len = marker_pos
        assert head_len > 1000  # at least ~50%
        assert head_len < 1500  # at most ~75%

    def test_bookend_tail_ratio(self):
        """Tail portion should be ~20% of max_chars."""
        lines = [f"line-{i:04d}" for i in range(500)]
        text = "\n".join(lines)

        result = truncate_tool_output(text, max_chars=2000)

        marker_end = result.rindex("chars]...") + len("chars]...")
        tail_len = len(result) - marker_end
        assert tail_len > 200  # at least ~10%
        assert tail_len < 600  # at most ~30%

    def test_empty_text_unchanged(self):
        """Empty text should be returned unchanged."""
        result = truncate_tool_output("", max_chars=2000)
        assert result == ""

    def test_none_text_returns_empty(self):
        """None text should return empty string."""
        result = truncate_tool_output(None, max_chars=2000)
        assert result == ""

    def test_truncation_marker_reports_dropped_count(self):
        """The truncation marker should report how many chars were dropped."""
        text = "a" * 5000
        result = truncate_tool_output(text, max_chars=2000)

        # Extract the number from the marker
        import re

        match = re.search(r"\[truncated (\d+) chars\]", result)
        assert match is not None
        dropped = int(match.group(1))
        # We started with 5000 chars, max is 2000, so should have dropped ~3000
        assert dropped > 2500
        assert dropped < 3500

    def test_head_strategy(self):
        """Strategy 'head' should just keep the first max_chars."""
        text = "AAAA" * 500 + "BBBB" * 500
        result = truncate_tool_output(text, max_chars=2000, strategy="head")

        assert result.startswith("AAAA")
        assert "BBBB" not in result
        assert "...[truncated" in result

    def test_default_strategy_is_bookend(self):
        """Default strategy should be bookend."""
        text = "HEAD" + "x" * 5000 + "TAIL"
        result = truncate_tool_output(text, max_chars=2000)

        assert "HEAD" in result
        assert "TAIL" in result

    def test_very_small_max_chars(self):
        """Very small max_chars should still work without crashing."""
        text = "a" * 1000
        result = truncate_tool_output(text, max_chars=50)
        # Should not crash, and should be reasonable
        assert len(result) < 200  # some tolerance

    def test_multiline_text_splits_cleanly(self):
        """Truncation should happen at line boundaries when possible."""
        lines = [f"line-{i}" for i in range(200)]
        text = "\n".join(lines)

        result = truncate_tool_output(text, max_chars=500)

        # Head should end at a newline boundary
        marker_pos = result.index("...[truncated")
        head = result[:marker_pos].rstrip("\n")
        # Each visible line should be complete (no partial lines)
        for line in head.split("\n"):
            assert line.startswith("line-")
