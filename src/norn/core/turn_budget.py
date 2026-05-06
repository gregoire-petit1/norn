"""Per-turn output budget tracker (Phase 10 Feature 2).

Caps total chars of tool output injected per LLM round. When multiple
tool calls would exceed the budget, applies progressive truncation.
"""

from __future__ import annotations

from norn.core.truncation import truncate_tool_output


class TurnBudgetTracker:
    """Track and enforce per-turn tool output budget.

    Usage:
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        # At start of each LLM round:
        tracker.reset()
        # For each tool result:
        content = tracker.allocate(raw_output, max_per_tool=8000)
    """

    def __init__(self, max_chars_per_turn: int = 30_000) -> None:
        self.max_chars_per_turn = max_chars_per_turn
        self._chars_used = 0

    @property
    def remaining(self) -> int:
        """Characters remaining in this turn's budget."""
        return max(0, self.max_chars_per_turn - self._chars_used)

    def reset(self) -> None:
        """Reset budget at the start of each LLM round."""
        self._chars_used = 0

    def allocate(self, raw_output: str, max_per_tool: int) -> str:
        """Allocate budget for a tool result, applying truncation if needed.

        Args:
            raw_output: The raw tool output text.
            max_per_tool: Per-tool character limit (existing config).

        Returns:
            Truncated output fitting within both per-tool and remaining budget.
        """
        remaining = self.remaining
        if remaining <= 0:
            return "[Output budget exhausted for this turn]"

        effective_max = min(max_per_tool, remaining)
        result = truncate_tool_output(raw_output, effective_max)
        self._chars_used += len(result)
        return result
