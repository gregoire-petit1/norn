"""Tool output truncation utilities (W1.2).

Provides strategies for truncating large tool outputs before they are
appended to the conversation history, reducing token consumption.
"""

from __future__ import annotations

from typing import Literal

# Ratio of max_chars allocated to the head portion in bookend strategy.
_BOOKEND_HEAD_RATIO = 0.60
# Ratio allocated to the tail portion.
_BOOKEND_TAIL_RATIO = 0.20


def truncate_tool_output(
    text: str | None,
    max_chars: int,
    *,
    strategy: Literal["bookend", "head"] = "bookend",
) -> str:
    """Truncate tool output text if it exceeds *max_chars*.

    Args:
        text: The tool output text. None is treated as empty string.
        max_chars: Maximum character count before truncation triggers.
        strategy: ``"bookend"`` keeps first 60% and last 20%;
            ``"head"`` keeps only the first portion.

    Returns:
        The (possibly truncated) text.
    """
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text

    if strategy == "head":
        return _truncate_head(text, max_chars)
    return _truncate_bookend(text, max_chars)


def _truncate_bookend(text: str, max_chars: int) -> str:
    """Keep first ~60% and last ~20% of the text, joined by a marker."""
    head_budget = int(max_chars * _BOOKEND_HEAD_RATIO)
    tail_budget = int(max_chars * _BOOKEND_TAIL_RATIO)

    head = _snap_to_line(text, head_budget, from_end=False)
    tail = _snap_to_line(text, tail_budget, from_end=True)

    dropped = len(text) - len(head) - len(tail)
    marker = f"\n...[truncated {dropped} chars]...\n"

    return head + marker + tail


def _truncate_head(text: str, max_chars: int) -> str:
    """Keep only the first portion."""
    head = _snap_to_line(text, max_chars - 60, from_end=False)  # reserve space for marker
    dropped = len(text) - len(head)
    marker = f"\n...[truncated {dropped} chars]..."
    return head + marker


def _snap_to_line(text: str, budget: int, *, from_end: bool) -> str:
    """Extract a portion of text, snapping to line boundaries.

    Args:
        text: Source text.
        budget: Approximate character budget.
        from_end: If True, take from the end of text; else from the start.

    Returns:
        A substring snapped to the nearest newline within budget.
    """
    if budget <= 0:
        return ""

    if from_end:
        raw = text[-budget:]
        # Snap forward to next newline to avoid partial lines
        nl = raw.find("\n")
        if nl != -1 and nl < len(raw) - 1:
            return raw[nl + 1 :]
        return raw

    raw = text[:budget]
    # Snap backward to last newline to avoid partial lines
    nl = raw.rfind("\n")
    if nl != -1:
        return raw[: nl + 1]
    return raw
