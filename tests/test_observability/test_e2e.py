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
    completion_fn: Any = None,
):
    """Build an AgentLoop whose LLM is a real RouterProvider wrapping LiteLLMProvider(s).

    Only ``litellm.acompletion`` needs mocking at call sites. A real
    ``PermissionChecker`` is attached so ``permission.decision`` events fire.

    If ``completion_fn`` is provided, it is injected per-instance into every
    tier's ``LiteLLMProvider``. This is preferable to ``patch("litellm.acompletion")``
    in concurrent tests, because the latter mutates module state and cross-
    contaminates tasks scheduled in parallel.
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

    if completion_fn is not None:
        # Override every tier's per-instance completion hook. Each tier provider
        # is independent, so concurrent tasks using different `completion_fn`s
        # cannot collide.
        for provider in router._providers.values():
            provider._completion_fn = completion_fn

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

    # First tier 'fast' fails with a NON-transient technical error (malformed
    # response) -> router falls back to the next tier. A transient 5xx/connection
    # error would instead be retried within the same tier by the LLM layer
    # (see test_llm transient-retry tests), so we use tool_use_failed here to
    # exercise the router fallback path specifically.
    # Second tier 'standard' returns a final response (no tool calls, keep test simple).
    ok = _make_llm_response(content="recovered", tool_calls=None)
    acompletion = AsyncMock(
        side_effect=[RuntimeError("finish_reason tool_use_failed"), ok],
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
    assert "tool_use_failed" in fb["error_message"]

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
        # Cost key must be present and numeric for a model with a known
        # litellm price entry (gpt-3.5-turbo). A None value would mean
        # the cost processor silently failed — see Phase 9 follow-up D2.
        assert "cost_usd" in ev, "cost_usd key missing when include_cost=True"
        assert isinstance(ev["cost_usd"], (int, float)), (
            f"cost_usd must be numeric for a priced model, got {type(ev['cost_usd']).__name__}"
        )
        assert ev["cost_usd"] > 0, (
            f"cost_usd must be positive for gpt-3.5-turbo with non-zero usage, got {ev['cost_usd']}"
        )
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
        # Per-task per-instance mocks: each `LiteLLMProvider` gets its own
        # `completion_fn`. Unlike `patch("litellm.acompletion", ...)` (which
        # mutates the module global and is non-deterministic under concurrent
        # tasks), this is immune to scheduling order.
        completion_mock_a = AsyncMock(
            return_value=_make_llm_response(content="done-A", tool_calls=None),
        )
        completion_mock_b = AsyncMock(
            return_value=_make_llm_response(content="done-B", tool_calls=None),
        )

        async def _one_run(label: str, completion_mock: AsyncMock) -> str:
            # Each task's new_session() mutates only its own copied context
            # (asyncio.Task copies the caller's context on creation); the
            # write stays local to the task.
            sid = new_session()
            agent = _make_router_agent(
                permission_mode=PermissionMode.AUTO,
                completion_fn=completion_mock,
            )
            await agent.run(f"task-{label}")
            return sid

        # Create tasks explicitly so each gets its own copied Context snapshot.
        t_a = asyncio.create_task(_one_run("A", completion_mock_a))
        t_b = asyncio.create_task(_one_run("B", completion_mock_b))
        sid_a, sid_b = await asyncio.gather(t_a, t_b)

        assert sid_a != sid_b

        # Isolation proof: each task hit *its own* mock exactly once. If the
        # injection were leaky (e.g. via a shared module-global patch), one
        # mock would absorb both calls.
        completion_mock_a.assert_awaited_once()
        completion_mock_b.assert_awaited_once()

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


# --------------------------------------------------------------------------- #
# D3.2 - denied destructive command path
# --------------------------------------------------------------------------- #


class _BashInput(BaseModel):
    command: str


class _BashLikeTool:
    """A HIGH-risk tool named ``bash`` so the RiskClassifier can flag
    destructive commands and trigger a permission denial under STRICT mode.

    The tool's body is never reached in the denied-path test — the
    PermissionChecker rejects the call before execute() is invoked.
    """

    name: str = "bash"
    description: str = "Execute a shell command (mock)."
    risk_level: RiskLevel = RiskLevel.HIGH
    input_model: type[BaseModel] = _BashInput

    async def execute(self, input: _BashInput, ctx: Any) -> ToolResult:  # noqa: A002
        return ToolResult(output=f"ran:{input.command}")


def _make_router_agent_with_bash(
    *,
    permission_mode,
    completion_fn: Any = None,
):
    """Variant of ``_make_router_agent`` with a HIGH-risk ``bash`` tool
    instead of the LOW-risk echo tool."""
    from norn.core.agent import AgentLoop
    from norn.core.router import RouterProvider
    from norn.tools.registry import ToolRegistry

    cfg = RouterConfig(tiers={"fast": RouterTierConfig(provider="openai", model="model-fast")})
    router = RouterProvider(cfg)
    if completion_fn is not None:
        for provider in router._providers.values():
            provider._completion_fn = completion_fn

    registry = ToolRegistry()
    registry.register(_BashLikeTool())

    checker = PermissionChecker(mode=permission_mode, classifier=RiskClassifier())
    return AgentLoop(llm=router, registry=registry, permission_checker=checker)


@pytest.mark.asyncio
async def test_denied_destructive_command_logs_permission_denied(log_dir: Path) -> None:
    """E2E: a destructive ``rm -rf`` under STRICT permission mode is denied
    by the classifier; ``tool.call`` records ``error_type=PermissionDenied``
    and ``permission.decision`` carries ``reason=destructive_denied``. The
    agent loop continues and produces a clean final answer.
    """
    agent = _make_router_agent_with_bash(permission_mode=PermissionMode.STRICT)

    # Turn 1: model asks to rm -rf (destructive). Turn 2: it acknowledges denial.
    first = _make_llm_response(
        content=None,
        tool_calls=[
            _make_tool_call_chunk(
                call_id="c1",
                tool_name="bash",
                arguments='{"command": "rm -rf /tmp/foo"}',
            )
        ],
        finish_reason="tool_calls",
    )
    second = _make_llm_response(content="Understood, command was denied.", tool_calls=None)

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first, second]),
    ):
        result = await agent.run("please rm -rf")

    assert result.content == "Understood, command was denied."

    events = _events(_today_file(log_dir))

    tool_calls = [e for e in events if e.get("event") == "tool.call"]
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["success"] is False
    assert tc["error_type"] == "PermissionDenied"
    assert "Permission denied" in tc["error_message"]

    perm_decisions = [e for e in events if e.get("event") == "permission.decision"]
    assert len(perm_decisions) == 1
    pd = perm_decisions[0]
    assert pd["granted"] is False
    assert pd["reason"] == "destructive_denied"
    assert pd["tool_name"] == "bash"

    # Agent loop completed cleanly: exactly one agent.run start + end with success=True.
    agent_events = [e for e in events if e.get("event") == "agent.run"]
    phases = [e["phase"] for e in agent_events]
    assert phases == ["start", "end"]
    assert agent_events[1]["success"] is True


# --------------------------------------------------------------------------- #
# D3.3 - agent.run failure: provider raises mid-flight
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_run_failure_emits_session_end_and_propagates(log_dir: Path) -> None:
    """E2E: when the LLM provider raises mid-flight, ``agent.run`` propagates
    the exception unchanged AND the lifecycle event is closed in the log
    with ``success=False`` plus the harmonized ``error_type`` /
    ``error_message`` pair.

    Contract verified:
    - exception is re-raised by ``measure_and_log`` (caller sees the failure)
    - ``agent.run`` end event carries ``success=False`` and the exception
      class name in ``error_type`` (so dashboards can filter failed sessions)
    - the start marker is still present (no "amputated" sessions in the log)
    """
    agent = _make_router_agent(permission_mode=PermissionMode.AUTO)

    # The router catches transient/technical errors for fallback, so we use
    # a single tier and a non-RuntimeError to ensure the exception bubbles
    # all the way out of router.complete -> agent.run.
    class _ProviderError(Exception):
        pass

    boom = AsyncMock(side_effect=_ProviderError("provider 500"))

    with (
        patch("litellm.acompletion", new=boom),
        pytest.raises(_ProviderError, match="provider 500"),
    ):
        await agent.run("trigger failure")

    events = _events(_today_file(log_dir))

    # Lifecycle is closed with a failure end event (not silently dropped).
    agent_events = [e for e in events if e.get("event") == "agent.run"]
    phases = [e.get("phase") for e in agent_events]
    assert phases == ["start", "end"], (
        f"agent.run lifecycle must be start+end even on failure, got {phases}"
    )
    end_ev = agent_events[1]
    assert end_ev["success"] is False
    assert end_ev["error_type"] == "_ProviderError"
    assert "provider 500" in end_ev["error_message"]
    assert isinstance(end_ev["duration_ms"], int) and end_ev["duration_ms"] >= 0


# --------------------------------------------------------------------------- #
# D3.4 - multi-tool turn: a single LLM response with N tool_calls
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multi_tool_turn_executes_in_order_and_logs_separately(
    log_dir: Path,
) -> None:
    """E2E: a single LLM turn returning 2 tool_calls runs them sequentially
    in arrival order and emits 2 distinct ``tool.call`` events (one per
    sub-call), each tagged with its own ``tool_name`` and ``success=True``.
    """
    agent = _make_router_agent(permission_mode=PermissionMode.AUTO)

    # Turn 1: two echo calls in a single tool_calls list.
    # Turn 2: final text answer.
    first = _make_llm_response(
        content=None,
        tool_calls=[
            _make_tool_call_chunk(call_id="c1", tool_name="echo", arguments='{"msg": "alpha"}'),
            _make_tool_call_chunk(call_id="c2", tool_name="echo", arguments='{"msg": "beta"}'),
        ],
        finish_reason="tool_calls",
    )
    second = _make_llm_response(content="both done", tool_calls=None)

    with patch(
        "litellm.acompletion",
        new=AsyncMock(side_effect=[first, second]),
    ):
        result = await agent.run("call echo twice")

    assert result.content == "both done"

    events = _events(_today_file(log_dir))

    tool_calls = [e for e in events if e.get("event") == "tool.call"]
    assert len(tool_calls) == 2, (
        f"expected 2 tool.call events from a single multi-tool turn, got {len(tool_calls)}"
    )
    # Both succeeded (no error_type emitted on success path).
    for tc in tool_calls:
        assert tc["tool_name"] == "echo"
        assert tc["success"] is True
        assert "error_type" not in tc

    # Ordering: both tool.call events are emitted between the single
    # llm.complete (turn 1, finish_reason=tool_calls) and the final
    # llm.complete (turn 2). There must be exactly 2 llm.complete events.
    llm_events = [e for e in events if e.get("event") == "llm.complete"]
    assert len(llm_events) == 2, f"expected 2 llm.complete (1 per turn), got {len(llm_events)}"

    stream = [(i, e.get("event")) for i, e in enumerate(events)]
    llm_idx = [i for i, n in stream if n == "llm.complete"]
    tool_idx = [i for i, n in stream if n == "tool.call"]
    # Both tool.call events fall between the two llm.complete events.
    assert llm_idx[0] < tool_idx[0] < tool_idx[1] < llm_idx[1], (
        f"expected llm1 < tool1 < tool2 < llm2, got llm={llm_idx} tool={tool_idx}"
    )

    # Permission was checked once per sub-call (LOW risk → auto_approved each time).
    perm_decisions = [e for e in events if e.get("event") == "permission.decision"]
    assert len(perm_decisions) == 2
    for pd in perm_decisions:
        assert pd["granted"] is True
        assert pd["tool_name"] == "echo"
