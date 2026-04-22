"""LLM-as-Judge: evaluates agent output quality using an LLM."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider

from benchmarks.runner.models import JudgeScore, TaskDef
from norn.core.models import Message, Role


JUDGE_SYSTEM_PROMPT = """\
You are an expert code reviewer evaluating an AI coding agent's output.
Score the output on three dimensions (0-10 scale each):

1. **correctness**: Does the code work correctly? Does it solve the stated problem?
2. **quality**: Is the code clean, readable, and well-structured?
3. **completeness**: Does it fully address all requirements?

Respond ONLY with a JSON object (no markdown, no explanation outside JSON):
{
    "correctness": <float 0-10>,
    "quality": <float 0-10>,
    "completeness": <float 0-10>,
    "reasoning": "<brief explanation>"
}
"""


def build_judge_prompt(task: TaskDef, agent_output: str) -> str:
    """Build the user prompt for the judge LLM."""
    return (
        f"## Task\n"
        f"**ID:** {task.id}\n"
        f"**Category:** {task.category}\n"
        f"**Description:** {task.description}\n"
        f"**Prompt given to agent:** {task.prompt}\n\n"
        f"## Agent Output\n"
        f"```\n{agent_output}\n```\n\n"
        f"Score this output. Respond with JSON only."
    )


def _extract_json(text: str) -> dict[str, Any] | None:
    """Extract JSON from text, handling markdown code blocks."""
    # Try direct parse first
    try:
        return json.loads(text.strip())
    except (json.JSONDecodeError, ValueError):
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except (json.JSONDecodeError, ValueError):
            pass

    return None


async def judge_task(
    task: TaskDef,
    agent_output: str,
    *,
    llm: LLMProvider,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> JudgeScore | None:
    """Use an LLM to judge the agent's output for a task.

    Returns JudgeScore on success, None on any failure (malformed response,
    missing fields, LLM error).
    """
    messages = [
        Message(role=Role.SYSTEM, content=JUDGE_SYSTEM_PROMPT),
        Message(role=Role.USER, content=build_judge_prompt(task, agent_output)),
    ]

    try:
        response = await llm.complete(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception:
        return None

    if not response.content:
        return None

    data = _extract_json(response.content)
    if data is None:
        return None

    try:
        return JudgeScore.model_validate(data)
    except Exception:
        return None
