"""Tests for per-turn output budget tracker."""

from __future__ import annotations

from norn.core.turn_budget import TurnBudgetTracker


class TestTurnBudgetTracker:
    """Test TurnBudgetTracker budget allocation."""

    def test_first_call_gets_full_budget(self):
        """First tool call gets min(per_tool, per_turn) budget."""
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        text = "x" * 5000
        result = tracker.allocate(text, max_per_tool=8000)
        assert result == text  # under both limits

    def test_large_output_truncated_by_per_tool(self):
        """Output exceeding per-tool limit is truncated."""
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        text = "x" * 10000
        result = tracker.allocate(text, max_per_tool=8000)
        assert len(result) <= 8000

    def test_budget_depletes_across_calls(self):
        """Budget decreases with each allocation."""
        tracker = TurnBudgetTracker(max_chars_per_turn=10000)
        # First call: 6000 chars
        text1 = "a" * 6000
        r1 = tracker.allocate(text1, max_per_tool=8000)
        assert r1 == text1
        # Second call: 6000 chars but only 4000 remaining
        text2 = "b" * 6000
        r2 = tracker.allocate(text2, max_per_tool=8000)
        assert len(r2) <= 4000

    def test_budget_exhausted_returns_marker(self):
        """When budget is fully exhausted, returns marker message."""
        tracker = TurnBudgetTracker(max_chars_per_turn=100)
        # Exhaust budget
        tracker.allocate("x" * 100, max_per_tool=200)
        # Next call should get marker
        result = tracker.allocate("y" * 500, max_per_tool=8000)
        assert "budget exhausted" in result.lower()

    def test_reset_restores_budget(self):
        """reset() restores full budget for next turn."""
        tracker = TurnBudgetTracker(max_chars_per_turn=100)
        tracker.allocate("x" * 100, max_per_tool=200)
        tracker.reset()
        # Now full budget available again
        text = "y" * 50
        result = tracker.allocate(text, max_per_tool=8000)
        assert result == text

    def test_remaining_property(self):
        """remaining reports chars left in budget."""
        tracker = TurnBudgetTracker(max_chars_per_turn=10000)
        assert tracker.remaining == 10000
        tracker.allocate("x" * 3000, max_per_tool=8000)
        assert tracker.remaining == 7000
