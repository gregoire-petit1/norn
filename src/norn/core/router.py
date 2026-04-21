"""RouterProvider — 3-tier LLM routing with complexity-based selection and fallback."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.core.models import Message


_COMPLEX_KEYWORDS = frozenset({"architect", "design", "refactor", "optimize", "analyze", "compare"})


class Tier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


def _classify_complexity(messages: list[Message], tools: list[dict]) -> Tier:
    """Score message complexity and return the appropriate tier.

    Heuristic signals (each contributes +1 to score):
    - Last message content longer than 500 chars
    - More than 10 messages in history
    - Complex keyword in last message (architect, design, refactor, optimize, analyze, compare)
    - More than 8 tools loaded

    Score 0 → FAST, 1-2 → STANDARD, 3+ → POWERFUL.
    """
    score = 0

    # Signal 1: long last message
    last = messages[-1] if messages else None
    if last and last.content and len(last.content) > 500:
        score += 1

    # Signal 2: long conversation history
    if len(messages) > 10:
        score += 1

    # Signal 3: complex keywords in last message
    if last and last.content:
        words = last.content.lower().split()
        if _COMPLEX_KEYWORDS.intersection(words):
            score += 1

    # Signal 4: many tools loaded
    if len(tools) > 8:
        score += 1

    if score == 0:
        return Tier.FAST
    if score <= 2:
        return Tier.STANDARD
    return Tier.POWERFUL
