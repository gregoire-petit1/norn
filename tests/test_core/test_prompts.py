"""Tests for system prompt (W1.3)."""

from norn.core.prompts import AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_is_compact():
    """System prompt should stay roughly bounded (~1200 chars)."""
    assert len(AGENT_SYSTEM_PROMPT) < 1200
    assert len(AGENT_SYSTEM_PROMPT) > 50  # not empty


def test_agent_system_prompt_emphasises_task_completion():
    """Prompt must tell the agent to finish *all* parts a task asks for.

    Regression guard: in run 7, long-horizon-001 wrote ``app.py`` but no
    tests because the prompt did not explicitly require finishing every
    part of the request.
    """
    lower = AGENT_SYSTEM_PROMPT.lower()
    assert "test" in lower
    assert any(kw in lower for kw in ("finish", "complete", "all parts"))


def test_agent_system_prompt_contains_identity():
    """System prompt should identify the agent as Norn."""
    assert "Norn" in AGENT_SYSTEM_PROMPT


def test_agent_system_prompt_mentions_tools():
    """System prompt should mention tool usage."""
    assert "tool" in AGENT_SYSTEM_PROMPT.lower()
