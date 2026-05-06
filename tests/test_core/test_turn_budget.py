"""Tests for per-turn output budget tracker."""

from __future__ import annotations

import pytest

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


class TestAgentIntegration:
    """Test budget tracker integration with AgentLoop."""

    @pytest.mark.asyncio
    async def test_agent_respects_turn_budget(self):
        """AgentLoop uses turn budget to cap multi-tool output."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop
        from norn.core.models import LLMResponse, ToolCall
        from norn.tools.base import ToolResult

        mock_llm = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        # First LLM call returns 3 tool calls, second returns text
        tool_calls = [
            ToolCall(id="t1", name="bash", arguments={"command": "echo hi"}),
            ToolCall(id="t2", name="bash", arguments={"command": "echo bye"}),
            ToolCall(id="t3", name="bash", arguments={"command": "echo end"}),
        ]
        mock_llm.complete = AsyncMock(
            side_effect=[
                LLMResponse(content=None, tool_calls=tool_calls),
                LLMResponse(content="Done"),
            ]
        )

        large_output = "x" * 5000

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            max_turn_output_chars=5000,  # Budget: 5000 total
            max_tool_result_chars=6000,  # Per-tool: 6000
            env_bootstrap=False,
        )

        # Mock _execute_tool to return large output
        async def mock_execute(call):
            return ToolResult(output=large_output)

        loop._execute_tool = mock_execute  # type: ignore[method-assign]

        await loop.run("test")

        # With 5000 budget and 3 tools each producing 5K:
        # Tool 1: gets full 5000 (remaining = 0)
        # Tool 2: remaining is 0, gets "budget exhausted" marker
        # Tool 3: same
        budget_msgs = [
            m for m in loop.history if m.content and "budget exhausted" in m.content.lower()
        ]
        assert len(budget_msgs) >= 1
