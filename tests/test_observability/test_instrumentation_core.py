"""Integration tests: core engines emit expected structured events.

These tests verify that Task 5 instrumentation (AgentLoop.run,
LiteLLMProvider.complete, RouterProvider) emits the events defined in
``norn.observability.events.EventName`` with the expected payload fields.

Each test uses an isolated ``LoggingConfig(file_dir=tmp_path)``; the autouse
``conftest._redirect_default_log_dir`` fixture provides an additional safety
net for any test that accidentally instantiates a default config.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import init_logging, new_session

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _today_file(dir_: Path) -> Path:
    return dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture
def log_dir(tmp_path: Path) -> Iterator[Path]:
    """Initialise file-only logging into tmp_path and start a session."""
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    new_session()
    yield tmp_path
    # Release file handles so tmp_path cleanup on Windows/macOS is clean.
    import logging as _logging

    for h in list(_logging.getLogger().handlers):
        h.close()
        _logging.getLogger().removeHandler(h)


# --------------------------------------------------------------------------- #
# LiteLLMProvider
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_llm_provider_emits_llm_complete(log_dir: Path) -> None:
    from norn.core.llm import LiteLLMProvider
    from norn.core.models import Message, Role

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(
            message=MagicMock(content="hi", tool_calls=None),
            finish_reason="stop",
        ),
    ]
    fake_response.usage = MagicMock(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
    )

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)):
        provider = LiteLLMProvider(model="gpt-4")
        await provider.complete(messages=[Message(role=Role.USER, content="hi")])

    events = _events(_today_file(log_dir))
    llm_events = [e for e in events if e.get("event") == "llm.complete"]
    assert len(llm_events) == 1
    ev = llm_events[0]
    assert ev["prompt_tokens"] == 10
    assert ev["completion_tokens"] == 5
    assert ev["total_tokens"] == 15
    assert ev["model"] == "gpt-4"
    assert ev["finish_reason"] == "stop"
    assert isinstance(ev["latency_ms"], int)
    assert ev["latency_ms"] >= 0
    assert "provider" in ev


@pytest.mark.asyncio
async def test_llm_provider_emits_llm_complete_on_error(log_dir: Path) -> None:
    from norn.core.llm import LiteLLMProvider
    from norn.core.models import Message, Role

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=RuntimeError("boom")),
    ):
        provider = LiteLLMProvider(model="gpt-4")
        with pytest.raises(RuntimeError, match="boom"):
            await provider.complete(messages=[Message(role=Role.USER, content="hi")])

    events = _events(_today_file(log_dir))
    err_events = [
        e
        for e in events
        if e.get("event") == "llm.complete" and e.get("error_type") == "RuntimeError"
    ]
    assert len(err_events) == 1
    assert err_events[0]["error"] == "boom"
    assert err_events[0]["level"] == "error"


# --------------------------------------------------------------------------- #
# RouterProvider
# --------------------------------------------------------------------------- #


def _make_router(tier_behaviours: dict[str, object]):
    """Build a RouterProvider whose per-tier providers are replaced by mocks.

    ``tier_behaviours`` maps tier name ("fast"/"standard"/"powerful") to either
    an ``LLMResponse`` to return, or an ``Exception`` instance to raise.
    """
    from norn.core.config import RouterConfig, RouterTierConfig
    from norn.core.router import RouterProvider, Tier

    cfg = RouterConfig(
        tiers={
            name: RouterTierConfig(provider="openai", model=f"model-{name}")
            for name in tier_behaviours
        },
    )
    router = RouterProvider(cfg)

    for name, behaviour in tier_behaviours.items():
        tier = Tier(name)
        mock_provider = MagicMock()
        if isinstance(behaviour, Exception):
            mock_provider.complete = AsyncMock(side_effect=behaviour)
        else:
            mock_provider.complete = AsyncMock(return_value=behaviour)
        router._providers[tier] = mock_provider  # noqa: SLF001 — test override

    return router


@pytest.mark.asyncio
async def test_router_emits_routing_decision(log_dir: Path) -> None:
    from norn.core.models import LLMResponse, Message, Role, TokenUsage

    ok = LLMResponse(content="ok", usage=TokenUsage())
    router = _make_router({"fast": ok, "standard": ok, "powerful": ok})

    await router.complete(messages=[Message(role=Role.USER, content="hi")])

    events = _events(_today_file(log_dir))
    decisions = [e for e in events if e.get("event") == "routing.decision"]
    assert len(decisions) == 1
    ev = decisions[0]
    assert ev["chosen_tier"] in {"fast", "standard", "powerful"}
    assert "score" in ev
    assert "signals" in ev


@pytest.mark.asyncio
async def test_router_emits_fallback_on_error(log_dir: Path) -> None:
    from norn.core.models import LLMResponse, Message, Role, TokenUsage
    from norn.core.router import Tier

    ok = LLMResponse(content="ok", usage=TokenUsage())
    # A technical error on "fast" must trigger fallback to "standard".
    router = _make_router(
        {
            "fast": RuntimeError("http error 503 service unavailable"),
            "standard": ok,
        },
    )

    await router.complete(
        messages=[Message(role=Role.USER, content="hi")],
        tier_override=Tier.FAST,
    )

    events = _events(_today_file(log_dir))
    fallbacks = [e for e in events if e.get("event") == "fallback"]
    assert len(fallbacks) == 1
    ev = fallbacks[0]
    assert ev["from_tier"] == "fast"
    assert ev["to_tier"] == "standard"
    assert ev["error_type"] == "RuntimeError"
    assert "503" in ev["error_message"]
    assert ev["level"] == "warning"


# --------------------------------------------------------------------------- #
# AgentLoop
# --------------------------------------------------------------------------- #


def _make_agent(llm_response):
    from norn.core.agent import AgentLoop
    from norn.tools.registry import ToolRegistry

    llm = MagicMock()
    if isinstance(llm_response, Exception):
        llm.complete = AsyncMock(side_effect=llm_response)
    else:
        llm.complete = AsyncMock(return_value=llm_response)
    return AgentLoop(llm=llm, registry=ToolRegistry())


@pytest.mark.asyncio
async def test_agent_run_emits_start_and_end(log_dir: Path) -> None:
    from norn.core.models import LLMResponse, TokenUsage

    agent = _make_agent(LLMResponse(content="done", usage=TokenUsage()))
    await agent.run("hello")

    events = _events(_today_file(log_dir))
    agent_events = [e for e in events if e.get("event") == "agent.run"]
    assert len(agent_events) == 2
    phases = [e["phase"] for e in agent_events]
    assert phases == ["start", "end"]
    # Same session_id across both events.
    sids = {e.get("session_id") for e in agent_events}
    assert len(sids) == 1
    assert next(iter(sids)) is not None


@pytest.mark.asyncio
async def test_agent_run_records_duration(log_dir: Path) -> None:
    from norn.core.models import LLMResponse, TokenUsage

    agent = _make_agent(LLMResponse(content="done", usage=TokenUsage()))
    await agent.run("hello")

    events = _events(_today_file(log_dir))
    end_events = [e for e in events if e.get("event") == "agent.run" and e.get("phase") == "end"]
    assert len(end_events) == 1
    ev = end_events[0]
    assert isinstance(ev["duration_ms"], int)
    assert ev["duration_ms"] >= 0
    assert isinstance(ev["success"], bool)
    assert ev["success"] is True
    assert ev.get("error") is None


@pytest.mark.asyncio
async def test_agent_run_records_failure(log_dir: Path) -> None:
    agent = _make_agent(RuntimeError("llm exploded"))
    with pytest.raises(RuntimeError, match="llm exploded"):
        await agent.run("hello")

    events = _events(_today_file(log_dir))
    end_events = [e for e in events if e.get("event") == "agent.run" and e.get("phase") == "end"]
    assert len(end_events) == 1
    ev = end_events[0]
    assert ev["success"] is False
    assert "RuntimeError" in ev["error"]
