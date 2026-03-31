"""Tests for dream prompt templates."""

from norn.dream.prompts import build_consolidation_prompt, build_dream_system_prompt


def test_dream_system_prompt_is_string():
    prompt = build_dream_system_prompt()
    assert isinstance(prompt, str)
    assert len(prompt) > 100
    assert "memory" in prompt.lower()


def test_dream_system_prompt_has_rules():
    prompt = build_dream_system_prompt()
    assert "MEMORY.md" in prompt
    assert "200 lines" in prompt or "200" in prompt
    assert "absolute dates" in prompt.lower() or "relative dates" in prompt.lower()


def test_consolidation_prompt_includes_memory():
    prompt = build_consolidation_prompt(
        current_memory="# Norn Memory\n\n- User prefers Python",
        daily_logs={"2026-03-31": "Session 1: refactored auth"},
        topic_files={"auth": "# Auth\n\nUsing JWT."},
    )
    assert "User prefers Python" in prompt
    assert "refactored auth" in prompt
    assert "JWT" in prompt


def test_consolidation_prompt_handles_empty():
    prompt = build_consolidation_prompt(
        current_memory="",
        daily_logs={},
        topic_files={},
    )
    assert isinstance(prompt, str)
    assert "no existing memory" in prompt.lower() or "empty" in prompt.lower()


def test_consolidation_prompt_multiple_days():
    prompt = build_consolidation_prompt(
        current_memory="# Norn Memory",
        daily_logs={
            "2026-03-31": "Session 1: did X",
            "2026-03-30": "Session 2: did Y",
        },
        topic_files={},
    )
    assert "2026-03-31" in prompt
    assert "2026-03-30" in prompt
