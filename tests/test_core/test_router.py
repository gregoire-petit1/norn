"""Tests for RouterProvider."""

from __future__ import annotations

from norn.core.models import Message, Role
from norn.core.router import Tier, _classify_complexity


def _msg(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _history(n: int) -> list[Message]:
    return [_msg("msg") for _ in range(n)]


# ── Tier classification ────────────────────────────────────────────────────


def test_classify_simple_prompt_is_fast():
    messages = [_msg("hello")]
    assert _classify_complexity(messages, []) == Tier.FAST


def test_classify_long_prompt_is_standard():
    messages = [_msg("x" * 501)]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_long_history_is_standard():
    messages = _history(11)
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_keyword_in_prompt_is_standard():
    for kw in ("architect", "design", "refactor", "optimize", "analyze", "compare"):
        messages = [_msg(f"please {kw} this")]
        assert _classify_complexity(messages, []) == Tier.STANDARD, f"keyword={kw}"


def test_classify_keyword_case_insensitive():
    messages = [_msg("REFACTOR this module")]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_many_tools_is_standard():
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    messages = [_msg("hello")]
    assert _classify_complexity(messages, tools) == Tier.STANDARD


def test_classify_multiple_signals_is_powerful():
    # long prompt + keyword + long history = score 3 → POWERFUL
    messages = _history(11) + [_msg("x" * 501 + " please refactor")]
    assert _classify_complexity(messages, []) == Tier.POWERFUL


def test_classify_empty_messages_is_fast():
    assert _classify_complexity([], []) == Tier.FAST


def test_classify_score_boundary_standard():
    # score exactly 2: long history + many tools → STANDARD
    messages = _history(11)
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    assert _classify_complexity(messages, tools) == Tier.STANDARD


def test_classify_score_boundary_powerful():
    # score 3: long history + many tools + keyword → POWERFUL
    messages = _history(11) + [_msg("please analyze")]
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    assert _classify_complexity(messages, tools) == Tier.POWERFUL
