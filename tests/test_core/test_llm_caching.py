"""Tests for prompt-cache helpers and integration in LiteLLMProvider.

Phase 9 v2 — Workstream G.
"""

from __future__ import annotations

from norn.core.llm import _apply_cache_markers, _supports_prompt_cache

# --------------------------------------------------------------------------- #
# G.1 — _supports_prompt_cache eligibility
# --------------------------------------------------------------------------- #


def test_supports_prompt_cache_anthropic_direct():
    assert _supports_prompt_cache("anthropic/claude-sonnet-4")


def test_supports_prompt_cache_anthropic_via_openrouter():
    assert _supports_prompt_cache("openrouter/anthropic/claude-sonnet-4")


def test_supports_prompt_cache_openai_4o():
    assert _supports_prompt_cache("openai/gpt-4o")


def test_supports_prompt_cache_openai_41():
    assert _supports_prompt_cache("openai/gpt-4.1")


def test_supports_prompt_cache_unsupported_ollama():
    assert not _supports_prompt_cache("ollama/qwen2.5-coder:14b")


def test_supports_prompt_cache_unsupported_openrouter_other():
    assert not _supports_prompt_cache("openrouter/stepfun/step-3.5-flash:free")


# --------------------------------------------------------------------------- #
# G.1 — _apply_cache_markers behaviour
# --------------------------------------------------------------------------- #


def test_apply_cache_markers_disabled_is_noop():
    msgs = [{"role": "system", "content": "you are an agent"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    out_msgs, out_tools = _apply_cache_markers(msgs, tools, enabled=False)
    assert out_msgs == msgs
    assert out_tools == tools


def test_apply_cache_markers_tags_system_prompt():
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "hi"},
    ]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert isinstance(out_msgs[0]["content"], list)
    assert out_msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out_msgs[0]["content"][0]["text"] == "you are an agent"
    # User message untouched
    assert out_msgs[1] == msgs[1]


def test_apply_cache_markers_tags_last_tool():
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]
    _, out_tools = _apply_cache_markers([], tools, enabled=True)
    assert out_tools[0] == tools[0]  # untouched
    assert out_tools[1]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_markers_no_system_no_crash():
    msgs = [{"role": "user", "content": "hi"}]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert out_msgs == msgs


def test_apply_cache_markers_empty_messages():
    out_msgs, out_tools = _apply_cache_markers([], None, enabled=True)
    assert out_msgs == []
    assert out_tools is None
