"""Empty LLM response → nudge + retry instead of ending the turn (wave 3).

Reproduces tb2 circuit-fibsqrt on gpt-5.3-codex: finish_reason=stop with
0 completion tokens, 3 times, on all three Norn arms — each time Norn
treated the void as a final answer and abandoned the task. Terminus-2
passed the same task.
"""

from unittest.mock import AsyncMock

import pytest

from norn.core.agent import EMPTY_RESPONSE_MAX_RETRIES, EMPTY_RESPONSE_NUDGE, AgentLoop
from norn.core.models import LLMResponse, Role, StreamChunk
from norn.tools.registry import ToolRegistry


def _agent(llm):
    return AgentLoop(llm=llm, registry=ToolRegistry(), env_bootstrap=False, repo_map=False)


@pytest.mark.asyncio
async def test_run_retries_after_empty_then_uses_real_answer():
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(content=""),
            LLMResponse(content=None),
            LLMResponse(content="real answer"),
        ]
    )
    agent = _agent(llm)
    out = await agent.run("do it")
    assert out.content == "real answer"
    assert llm.complete.await_count == 3
    # the nudges are in the durable history (append-only), the empties are not
    nudges = [m for m in agent.history if m.role == Role.USER and m.content == EMPTY_RESPONSE_NUDGE]
    assert len(nudges) == 2


@pytest.mark.asyncio
async def test_run_gives_up_after_max_retries():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content=""))
    agent = _agent(llm)
    out = await agent.run("do it")
    # 1 initial + MAX retries, then the empty answer is accepted as final
    assert llm.complete.await_count == 1 + EMPTY_RESPONSE_MAX_RETRIES
    assert not (out.content or "").strip()


@pytest.mark.asyncio
async def test_run_non_empty_answer_not_retried():
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content="done"))
    agent = _agent(llm)
    await agent.run("do it")
    assert llm.complete.await_count == 1


async def _stream_of(*texts):
    for t in texts:
        yield StreamChunk(content=t, done=False)
    yield StreamChunk(content=None, done=True)


@pytest.mark.asyncio
async def test_run_stream_retries_after_empty():
    calls = {"n": 0}

    def _stream(**kw):
        calls["n"] += 1
        return _stream_of() if calls["n"] == 1 else _stream_of("real", " answer")

    llm = AsyncMock()
    llm.stream = _stream
    agent = _agent(llm)
    events = [ev async for ev in agent.run_stream("go")]
    text = "".join(e.content or "" for e in events if e.content)
    assert text == "real answer"
    assert calls["n"] == 2


@pytest.mark.asyncio
async def test_run_stream_gives_up_after_max_retries():
    calls = {"n": 0}

    def _stream(**kw):
        calls["n"] += 1
        return _stream_of()

    llm = AsyncMock()
    llm.stream = _stream
    agent = _agent(llm)
    async for _ in agent.run_stream("go"):
        pass
    assert calls["n"] == 1 + EMPTY_RESPONSE_MAX_RETRIES
