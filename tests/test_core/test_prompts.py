"""Tests for system prompt (W1.3)."""

from norn.core.prompts import AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_is_compact():
    """System prompt should be under 200 tokens (~800 chars)."""
    assert len(AGENT_SYSTEM_PROMPT) < 800
    assert len(AGENT_SYSTEM_PROMPT) > 50  # not empty


def test_agent_system_prompt_contains_identity():
    """System prompt should identify the agent as Norn."""
    assert "Norn" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_mentions_tools():
    """System prompt should mention tool usage."""
    assert "tool" in AGENT_SYSTEM_PROMPT.lower()
