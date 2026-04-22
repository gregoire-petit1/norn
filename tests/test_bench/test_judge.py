"""Tests for LLM-as-Judge evaluator."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from benchmarks.runner.judge import judge_task, build_judge_prompt
from benchmarks.runner.models import JudgeScore, TaskDef


def _make_task(**kwargs) -> TaskDef:
    defaults = dict(
        id="test-001",
        category="code-gen",
        description="Implement fibonacci",
        prompt="Write fib(n)",
        seed_dir="seed/",
        eval_command="echo ok",
    )
    defaults.update(kwargs)
    return TaskDef(**defaults)


def _mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLMResponse with given content."""
    resp = MagicMock()
    resp.content = content
    resp.usage = None
    resp.tool_calls = []
    return resp


@pytest.mark.asyncio
async def test_judge_valid_json_score():
    """Valid JSON response is parsed into JudgeScore."""
    task = _make_task()
    agent_output = "def fib(n): return n if n <= 1 else fib(n-1) + fib(n-2)"

    score_json = json.dumps(
        {
            "correctness": 8.5,
            "quality": 7.0,
            "completeness": 9.0,
            "reasoning": "Correct recursive implementation but no memoization.",
        }
    )

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response(score_json))

    score = await judge_task(task, agent_output, llm=mock_llm)
    assert score is not None
    assert score.correctness == 8.5
    assert score.quality == 7.0
    assert score.completeness == 9.0
    assert "recursive" in score.reasoning.lower() or len(score.reasoning) > 0


@pytest.mark.asyncio
async def test_judge_malformed_json_returns_none():
    """Malformed JSON response returns None (graceful failure)."""
    task = _make_task()

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response("this is not json"))

    score = await judge_task(task, "some output", llm=mock_llm)
    assert score is None


@pytest.mark.asyncio
async def test_judge_json_missing_fields_returns_none():
    """JSON missing required fields returns None."""
    task = _make_task()

    incomplete_json = json.dumps({"correctness": 8.0})  # missing quality, completeness, reasoning
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response(incomplete_json))

    score = await judge_task(task, "some output", llm=mock_llm)
    assert score is None


@pytest.mark.asyncio
async def test_judge_score_threshold():
    """Score threshold logic works correctly."""
    task = _make_task()

    # High score
    high_json = json.dumps(
        {"correctness": 9.0, "quality": 8.0, "completeness": 8.0, "reasoning": "Excellent"}
    )
    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response(high_json))

    score = await judge_task(task, "good output", llm=mock_llm)
    assert score is not None
    assert score.passes(threshold=7.0) is True

    # Low score
    low_json = json.dumps(
        {"correctness": 3.0, "quality": 4.0, "completeness": 2.0, "reasoning": "Poor"}
    )
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response(low_json))

    score = await judge_task(task, "bad output", llm=mock_llm)
    assert score is not None
    assert score.passes(threshold=7.0) is False


@pytest.mark.asyncio
async def test_judge_llm_exception_returns_none():
    """LLM exception returns None (graceful failure)."""
    task = _make_task()

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(side_effect=Exception("API error"))

    score = await judge_task(task, "some output", llm=mock_llm)
    assert score is None


@pytest.mark.asyncio
async def test_judge_json_in_markdown_code_block():
    """JSON wrapped in markdown code block is still parsed."""
    task = _make_task()

    score_json = json.dumps(
        {"correctness": 7.0, "quality": 7.0, "completeness": 7.0, "reasoning": "Adequate."}
    )
    wrapped = f"```json\n{score_json}\n```"

    mock_llm = AsyncMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_response(wrapped))

    score = await judge_task(task, "some output", llm=mock_llm)
    assert score is not None
    assert score.correctness == 7.0


def test_build_judge_prompt():
    """Judge prompt contains task description and agent output."""
    task = _make_task()
    prompt = build_judge_prompt(task, "def fib(n): pass")
    assert "fibonacci" in prompt.lower() or "fib" in prompt.lower()
    assert "def fib(n): pass" in prompt
