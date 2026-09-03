"""Tests for system prompt (W1.3)."""

from norn.core.prompts import AGENT_SYSTEM_PROMPT, SELF_VERIFY_PROMPT


def test_agent_system_prompt_is_compact():
    """System prompt should stay bounded (~1500 chars).

    Cap raised from 1200 → 1600 for two KIRA-informed guidance blocks tied to
    observed terminal-bench failures: deliverable env-robustness (openssl task's
    check script imported cryptography, present under the agent's python but not
    the grader's — refined to "prefer stdlib/CLI over pip deps") and the "no
    eyes/ears → use programmatic tools for media" rule (chess-from-image,
    gcode decode). The prompt is prompt-cached, so the per-turn cost is marginal.
    """
    # Bumped 1600 -> 1800 for the wave-2 A1 task_notes guidance line, then
    # -> 1900 for the batching rule (replaces 'one tool call per step', which
    # measured at 2.5x the LLM calls of Terminus-2 for no pass-rate gain).
    assert len(AGENT_SYSTEM_PROMPT) < 1900
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


def test_self_verify_prompt_ends_with_verdict_token():
    """Caller greps the last PASS/FAIL token, so the prompt must demand one."""
    assert "PASS" in SELF_VERIFY_PROMPT
    assert "FAIL" in SELF_VERIFY_PROMPT


def test_self_verify_prompt_demands_execution_proof():
    """Regression: false-PASS failures (gcode/openssl) came from rubber-stamping.
    The verify prompt must require proof-by-execution, not conviction."""
    lower = SELF_VERIFY_PROMPT.lower()
    assert "prove" in lower or "proven" in lower
    # env-robustness guard (openssl python vs python3)
    assert "python" in lower
    # anti-overfit guard (video-processing tuned to example)
    assert any(kw in lower for kw in ("real inputs", "changed values", "overfit"))


def test_plan_prompt_is_verdict_free():
    """PLAN_PROMPT must not carry a PASS/FAIL token (would fool _extract_verdict)."""
    from norn.core.agent import AgentLoop
    from norn.core.prompts import PLAN_PROMPT

    assert PLAN_PROMPT.strip()
    assert "Steps" in PLAN_PROMPT and "Files" in PLAN_PROMPT and "Test" in PLAN_PROMPT
    assert AgentLoop._extract_verdict(PLAN_PROMPT) == "unknown"
