"""Tier classification primitives for the RouterProvider (Phase 7)."""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.core.models import Message


# Tier-classification thresholds (tunable heuristics).
_LONG_PROMPT_CHARS = 500
_LONG_HISTORY_TURNS = 10
_MANY_TOOLS = 8
_STANDARD_MAX_SCORE = 2  # score <= this → STANDARD; above → POWERFUL

_KEYWORD_RE = re.compile(
    r"\b(?:architect|design|refactor|optimize|analyze|compare)\b",
    re.IGNORECASE,
)


class Tier(StrEnum):
    """Routing tier — FAST (cheap/quick), STANDARD (default), POWERFUL (complex tasks)."""

    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


def _classify_complexity(messages: list[Message], tools: list[dict]) -> Tier:
    """Score message complexity and return the appropriate tier.

    Heuristic signals (each contributes +1 to score):
    - Last message content longer than _LONG_PROMPT_CHARS (500) chars
    - More than _LONG_HISTORY_TURNS (10) messages in history
    - Complex keyword in last message (architect, design, refactor, optimize, analyze, compare)
    - More than _MANY_TOOLS (8) tools loaded

    Score 0 → FAST, 1..2 → STANDARD, 3+ → POWERFUL.
    """
    score = 0

    last_content = messages[-1].content if messages else None
    if last_content:
        if len(last_content) > _LONG_PROMPT_CHARS:
            score += 1
        if _KEYWORD_RE.search(last_content):
            score += 1

    if len(messages) > _LONG_HISTORY_TURNS:
        score += 1

    if len(tools) > _MANY_TOOLS:
        score += 1

    if score == 0:
        return Tier.FAST
    if score <= _STANDARD_MAX_SCORE:
        return Tier.STANDARD
    return Tier.POWERFUL
