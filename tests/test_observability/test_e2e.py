"""End-to-end observability tests: full agent.run -> 6 event types with correlated session.

These tests exercise the full Phase 8 event pipeline by orchestrating a real
``AgentLoop`` wrapping a ``RouterProvider`` that wraps a ``LiteLLMProvider``
(only ``litellm.acompletion`` is mocked), a real ``ToolRegistry`` with a trivial
tool, and a real ``PermissionChecker``. The JSONL sink produced on disk is the
single source of truth under assertion — no structlog private APIs are touched.

Coverage:

- #1 ``test_full_run_emits_all_core_events_with_correlated_session_id``
- #2 ``test_fallback_event_emitted_on_tier_failure``
- #3 ``test_cost_included_when_enabled``
- #4 ``test_cost_absent_when_disabled``
- #5 ``test_redaction_e2e`` (+ B2 regression guard: exact-match redaction)
- #6 ``test_disabled_logging_writes_nothing``
- #7 ``test_concurrent_agent_runs_do_not_cross_contaminate_sessions``
- #8 ``test_event_ordering_llm_before_tool_after_llm``
"""

from __future__ import annotations

import asyncio
import json
import logging as _logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from norn.core.config import (
    LoggingConfig,
    PermissionMode,
    RouterConfig,
    RouterTierConfig,
)
from norn.observability.logger import get_logger, init_logging, new_session, session_id_var
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.tools.base import RiskLevel, ToolResult

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _today_file(dir_: Path) -> Path:
    return dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"


def _events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close_handlers() -> None:
    """Release log file handles so tmp_path cleanup is clean on all OSes."""
    root = _logging.getLogger()
    for h in list(root.handlers):
        h.close()
        root.removeHandler(h)


@pytest.fixture
def log_dir(tmp_path: Path) -> Iterator[Path]:
    """File-only logging into tmp_path + fresh session id. include_cost disabled."""
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    new_session()
    yield tmp_path
    _close_handlers()


# --------------------------------------------------------------------------- #
# Mock LLM response builders
# --------------------------------------------------------------------------- #


def _make_tool_call_chunk(
    *,
    call_id: str = "c1",
    tool_name: str = "echo",
    arguments: str = '{"msg": "x"}',
) -> MagicMock:
    """Build a litellm-shaped tool_call object."""
    tc = MagicMock()
    tc.id = call_id
    tc.function = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = arguments
    return tc


def _make_llm_response(
    *,
    content: str | None = None,
    tool_calls: list | None = None,
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 5,
) -> MagicMock:
    """Build a litellm-shaped acompletion response."""
    resp = MagicMock()
    resp.choices = [
        MagicMock(
            message=MagicMock(content=content, tool_calls=tool_calls),
            finish_reason=finish_reason,
        ),
    ]
    resp.usage = MagicMock(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return resp


# --------------------------------------------------------------------------- #
# Minimal real tool
# --------------------------------------------------------------------------- #


class _EchoInput(BaseModel):
    msg: str = "hi"


class _EchoTool:
    """A trivial LOW-risk tool: returns the input ``msg`` verbatim.

    Duck-typed to the ``Tool`` Protocol — avoids the MagicMock-as-Tool issue
    in AgentLoop._execute_tool_inner (which calls ``tool.input_model(**args)``).
    """

    name: str = "echo"
    description: str = "Echo back the input message."
    risk_level: RiskLevel = RiskLevel.LOW
    input_model: type[BaseModel] = _EchoInput

    async def execute(self, input: _EchoInput, ctx: Any) -> ToolResult:  # noqa: A002
        return ToolResult(output=f"echo:{input.msg}")


# --------------------------------------------------------------------------- #
# AgentLoop factory using RouterProvider(LiteLLMProvider)
# --------------------------------------------------------------------------- #


def _make_router_agent(
    tiers: list[str] | None = None,
    *,
    permission_mode: PermissionMode = PermissionMode.AUTO,
):
    """Build an AgentLoop whose LLM is a real RouterProvider wrapping LiteLLMProvider(s).

    Only ``litellm.acompletion`` needs mocking at call sites. A real
    ``PermissionChecker`` is attached so ``permission.decision`` events fire.
    """
    from norn.core.agent import AgentLoop
    from norn.core.router import RouterProvider
    from norn.tools.registry import ToolRegistry

    tier_names = tiers or ["fast"]
    cfg = RouterConfig(
        tiers={
            name: RouterTierConfig(provider="openai", model=f"model-{name}") for name in tier_names
        },
    )
    router = RouterProvider(cfg)

    registry = ToolRegistry()
    registry.register(_EchoTool())

    checker = PermissionChecker(mode=permission_mode, classifier=RiskClassifier())

    return AgentLoop(llm=router, registry=registry, permission_checker=checker)


# --------------------------------------------------------------------------- #
# #1 - THE headline acceptance test
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_full_run_emits_all_core_events_with_correlated_session_id(
    tmp_path: Path,
) -> None:
    """A complete agent.run emits all 6 core event types, all sharing one session_id."""
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    sid = new_session()

    try:
        agent = _make_router_agent(permission_mode=PermissionMode.AUTO)

        # Turn 1: model asks to call the echo tool.
        # Turn 2: model returns final answer, no tool_calls.
        first = _make_llm_response(
            content=None,
            tool_calls=[_make_tool_call_chunk(call_id="c1", tool_name="echo")],
            finish_reason="tool_calls",
        )
        second = _make_llm_response(content="all done", tool_calls=None)

        with patch(
            "litellm.acompletion",
            new=AsyncMock(side_effect=[first, second]),
        ):
            result = await agent.run("do something")

        assert result.content == "all done"

        events = _events(_today_file(tmp_path))

        # Partition by event name.
        by_name: dict[str, list[dict]] = {}
        for ev in events:
            by_name.setdefault(ev.get("event", ""), []).append(ev)

        # agent.run: exactly one start + one end(success=True)
        agent_events = by_name.get("agent.run", [])
        phases = [e["phase"] for e in agent_events]
        assert phases == ["start", "end"], f"expected [start, end], got {phases}"
        end_ev = agent_events[1]
        assert end_ev["success"] is True
        assert isinstance(end_ev["duration_ms"], int) and end_ev["duration_ms"] >= 0

        # llm.complete: one per turn (>= 2)
        llm_events = by_name.get("llm.complete", [])
        assert len(llm_events) >= 2, f"expected >=2 llm.complete, got {len(llm_events)}"
        for ev in llm_events:
            for key in (
                "model",
                "provider",
                "prompt_tokens",
                "completion_tokens",
                "total_tokens",
                "latency_ms",
                "finish_reason",
            ):
                assert key in ev, f"llm.complete missing field {key}: {ev}"
            # B1 regression guard: tier contextvar must have propagated via
            # merge_contextvars processor (router wraps each call).
            assert ev.get("tier") == "fast", (
                "tier contextvar did not propagate to llm.complete — "
                "merge_contextvars processor regression?"
            )

        # routing.decision: one per router.complete call (>= 1)
        routing_events = by_name.get("routing.decision", [])
        assert len(routing_events) >= 1
        for ev in routing_events:
            assert "chosen_tier" in ev
            assert "score" in ev
            assert "signals" in ev

        # tool.call: exactly one (the echo call)
        tool_events = by_name.get("tool.call", [])
        assert len(tool_events) == 1
        tc = tool_events[0]
        assert tc["tool_name"] == "echo"
        assert tc["success"] is True
        assert isinstance(tc["duration_ms"], int) and tc["duration_ms"] >= 0

        # permission.decision: at least one (for the echo tool)
        perm_events = by_name.get("permission.decision", [])
        assert len(perm_events) >= 1
        pd = perm_events[0]
        assert pd["tool_name"] == "echo"
        assert pd["granted"] is True
        assert pd["mode"] == "auto"
        assert "risk_level" in pd

        # Session correlation: every event carries sid.
        for ev in events:
            assert ev.get("session_id") == sid, (
                f"session_id mismatch: expected {sid}, got {ev.get('session_id')} "
                f"on event {ev.get('event')}"
            )
    finally:
        _close_handlers()


# --------------------------------------------------------------------------- #
# #2 - fallback event
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fallback_event_emitted_on_tier_failure(log_dir: Path) -> None:
    """A technical error on the first tier triggers fallback to the next."""
    agent = _make_router_agent(
        tiers=["fast", "standard"],
        permission_mode=PermissionMode.AUTO,
    )

    # First tier 'fast' fails with a technical (503) error -> router falls back.
    # Second tier 'standard' returns a final response (no tool calls, keep test simple).
    ok = _make_llm_response(content="recovered", tool_calls=None)
    acompletion = AsyncMock(
        side_effect=[RuntimeError("http error 503 service unavailable"), ok],
    )

    with patch("litellm.acompletion", new=acompletion):
        from norn.core.router import Tier

        # Force start_tier=FAST to guarantee fallback ordering (fast -> standard).
        agent.llm.default_tier = Tier.FAST
        result = await agent.run("do something")

    assert result.content == "recovered"

    events = _events(_today_file(log_dir))
    fallbacks = [e for e in events if e.get("event") == "fallback"]
    assert len(fallbacks) == 1, f"expected 1 fallback event, got {len(fallbacks)}"
    fb = fallbacks[0]
    assert fb["from_tier"] == "fast"
    assert fb["to_tier"] == "standard"
    assert fb["error_type"] == "RuntimeError"
    assert "503" in fb["error_message"]

    # Outer agent.run recovered.
    end_events = [e for e in events if e.get("event") == "agent.run" and e.get("phase") == "end"]
    assert len(end_events) == 1
    assert end_events[0]["success"] is True


# --------------------------------------------------------------------------- #
# #3 - cost enabled
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cost_included_when_enabled(tmp_path: Path) -> None:
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=True,
    )
    init_logging(cfg)
    new_session()
    try:
        from norn.core.llm import LiteLLMProvider
        from norn.core.models import Message, Role

        resp = _make_llm_response(content="hi", prompt_tokens=10, completion_tokens=5)
        with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
            provider = LiteLLMProvider(model="gpt-3.5-turbo")
            await provider.complete(messages=[Message(role=Role.USER, content="hi")])

        events = _events(_today_file(tmp_path))
        llm_events = [e for e in events if e.get("event") == "llm.complete"]
        assert len(llm_events) == 1
        ev = llm_events[0]
        # Cost key must be present. Value may be a float (>=0) or None if
        # litellm.completion_cost failed/returned a non-coercible value
        # (processor is fail-open by contract — see processors.py:make_cost_processor).
        assert "cost_usd" in ev, "cost_usd key missing when include_cost=True"
        assert ev["cost_usd"] is None or isinstance(ev["cost_usd"], int | float)
    finally:
        _close_handlers()


# --------------------------------------------------------------------------- #
# #4 - cost disabled
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_cost_absent_when_disabled(tmp_path: Path) -> None:
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    new_session()
    try:
        from norn.core.llm import LiteLLMProvider
        from norn.core.models import Message, Role

        resp = _make_llm_response(content="hi", prompt_tokens=1, completion_tokens=1)
        with patch("litellm.acompletion", new=AsyncMock(return_value=resp)):
            provider = LiteLLMProvider(model="gpt-3.5-turbo")
            await provider.complete(messages=[Message(role=Role.USER, content="hi")])

        events = _events(_today_file(tmp_path))
        llm_events = [e for e in events if e.get("event") == "llm.complete"]
        assert len(llm_events) == 1
        assert "cost_usd" not in llm_events[0], "cost_usd must not appear when include_cost=False"
    finally:
        _close_handlers()


# --------------------------------------------------------------------------- #
# #5 - redaction + B2 regression (exact-match, not substring)
# --------------------------------------------------------------------------- #


def test_redaction_e2e(tmp_path: Path) -> None:
    """api_key redacted; model preserved; prompt_tokens/completion_tokens NOT redacted.

    B2 regression: default redact_keys include "token", but matching is
    case-insensitive EXACT on key name — so ``prompt_tokens`` / ``completion_tokens``
    (substring "token") must NOT be scrubbed.
    """
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        # Use the production default redact_keys (includes "token") to exercise
        # the B2 regression guard.
    )
    init_logging(cfg)
    new_session()
    try:
        get_logger("test").info(
            "test.event",
            api_key="sk-secret",
            model="gpt-4",
            prompt_tokens=42,
            completion_tokens=17,
            total_tokens=59,
        )
        _close_handlers()  # flush before read

        events = _events(_today_file(tmp_path))
        assert len(events) == 1
        ev = events[0]
        assert ev["api_key"] == "[REDACTED]"
        assert ev["model"] == "gpt-4"
        # B2 guard: substring-"token" keys preserved.
        assert ev["prompt_tokens"] == 42
        assert ev["completion_tokens"] == 17
        assert ev["total_tokens"] == 59
    finally:
        _close_handlers()


# --------------------------------------------------------------------------- #
# #6 - disabled logging: zero files
# --------------------------------------------------------------------------- #


def test_disabled_logging_writes_nothing(tmp_path: Path) -> None:
    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    try:
        get_logger("test").info("should.not.appear", foo="bar")
        get_logger("test").error("also.silenced")
    finally:
        _close_handlers()

    # tmp_path must contain zero files: the sink was never attached.
    assert list(tmp_path.iterdir()) == [], (
        f"disabled logging must produce no files, found: {list(tmp_path.iterdir())}"
    )


# --------------------------------------------------------------------------- #
# #7 - concurrent agent.run: session isolation via ContextVar propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_concurrent_agent_runs_do_not_cross_contaminate_sessions(
    tmp_path: Path,
) -> None:
    """Two asyncio tasks each set their own session_id; events must not mix."""
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    # Deliberately do NOT set a session at the outer scope: each task creates its own.

    try:

        async def _one_run(label: str) -> str:
            # Each task's new_session() mutates only its own copied context
            # (asyncio.Task copies the caller's context on creation); the
            # write stays local to the task.
            sid = new_session()
            agent = _make_router_agent(permission_mode=PermissionMode.AUTO)
            resp = _make_llm_response(content=f"done-{label}", tool_calls=None)
            with patch(
                "litellm.acompletion",
                new=AsyncMock(return_value=resp),
            ):
                await agent.run(f"task-{label}")
            return sid

        # Create tasks explicitly so each gets its own copied Context snapshot.
        t_a = asyncio.create_task(_one_run("A"))
        t_b = asyncio.create_task(_one_run("B"))
        sid_a, sid_b = await asyncio.gather(t_a, t_b)

        assert sid_a != sid_b

        events = _events(_today_file(tmp_path))
        sids_seen = {ev.get("session_id") for ev in events}
        # Only the two sids should appear (outer context had no session).
        assert sids_seen == {sid_a, sid_b}, f"expected {{sid_a, sid_b}}, saw {sids_seen}"

        # Each session's events form a self-contained subset: at least
        # agent.run(start+end) + llm.complete + routing.decision.
        for sid in (sid_a, sid_b):
            subset = [e for e in events if e.get("session_id") == sid]
            names = {e.get("event") for e in subset}
            assert "agent.run" in names
            assert "llm.complete" in names
            assert "routing.decision" in names
    finally:
        _close_handlers()
        # The outer scope's session_id was never set; nothing to reset beyond
        # the autouse fixture's teardown.
        session_id_var.set(None)


# --------------------------------------------------------------------------- #
# #8 - event ordering causality
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_event_ordering_llm_before_tool_after_llm(log_dir: Path) -> None:
    """Validate the temporal ordering of the event stream in a 1-tool-call run:

        agent.run(start)
        llm.complete (turn 1 -> tool_calls)
        tool.call (echo)
        llm.complete (turn 2 -> final answer)
        agent.run(end)

    permission.decision and routing.decision are emitted in the stream too;
    we only assert the *relative* ordering of the five anchor events above.
    """
    agent = _make_router_agent(permission_mode=PermissionMode.AUTO)

    first = _make_llm_response(
        content=None,
        tool_calls=[_make_tool_call_chunk(call_id="c1", tool_name="echo")],
        finish_reason="tool_calls",
    )
    second = _make_llm_response(content="final", tool_calls=None)

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first, second]),
    ):
        await agent.run("do something")

    events = _events(_today_file(log_dir))

    # Build a list of (index, event_name, phase_or_none) in emission order.
    stream = [(i, e.get("event"), e.get("phase")) for i, e in enumerate(events)]

    def _idx(name: str, phase: str | None = None) -> int:
        for i, n, p in stream:
            if n == name and (phase is None or p == phase):
                return i
        raise AssertionError(f"missing event {name}(phase={phase}) in stream: {stream}")

    def _all_indices(name: str) -> list[int]:
        return [i for i, n, _ in stream if n == name]

    i_start = _idx("agent.run", phase="start")
    i_end = _idx("agent.run", phase="end")
    i_tool = _idx("tool.call")
    llm_indices = _all_indices("llm.complete")

    assert len(llm_indices) == 2, f"expected 2 llm.complete events, got {llm_indices}"
    i_llm1, i_llm2 = llm_indices

    # agent.run(start) is first.
    assert i_start == 0, f"agent.run(start) must be first, got index {i_start}"
    # agent.run(end) is last.
    assert i_end == len(stream) - 1, (
        f"agent.run(end) must be last, got index {i_end} of {len(stream)}"
    )
    # llm(turn1) < tool.call < llm(turn2).
    assert i_llm1 < i_tool < i_llm2, (
        f"expected llm1 < tool < llm2, got {i_llm1}, {i_tool}, {i_llm2}"
    )
    # And both bracketed by agent.run start/end.
    assert i_start < i_llm1
    assert i_llm2 < i_end
