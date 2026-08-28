"""Tests for the append-only thread invariant (SOTA v2, workstream A — W1.2)."""

import pytest

from norn.core.models import Message, Role
from norn.core.thread_ledger import (
    ThreadInvariantError,
    ThreadLedger,
    canonical_message_digest,
)


def _msg(role: Role = Role.USER, content: str = "hi") -> Message:
    return Message(role=role, content=content)


# --------------------------------------------------------------------------- #
# Digest chaining
# --------------------------------------------------------------------------- #


def test_digest_is_deterministic():
    m = _msg()
    assert canonical_message_digest(m) == canonical_message_digest(m)


def test_digest_chains_on_previous():
    m = _msg()
    assert canonical_message_digest(m, "") != canonical_message_digest(m, "prev")


def test_extend_is_incremental():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a")]
    ledger.extend(history)
    assert len(ledger) == 1
    history.append(_msg(content="b"))
    ledger.extend(history)
    assert len(ledger) == 2
    # Extending with no new messages is a no-op
    ledger.extend(history)
    assert len(ledger) == 2


# --------------------------------------------------------------------------- #
# verify() — replacement / truncation / reorder
# --------------------------------------------------------------------------- #


def test_verify_clean_history_passes():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a"), _msg(content="b")]
    ledger.extend(history)
    assert ledger.verify(history) == []


def test_verify_detects_replacement():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a"), _msg(content="b")]
    ledger.extend(history)
    history[0] = _msg(content="tampered")
    violations = ledger.verify(history)
    assert violations and "replaced" in violations[0]


def test_verify_accepts_equal_replacement_object():
    """A new object with identical content re-hashes to the same digest."""
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a")]
    ledger.extend(history)
    history[0] = _msg(content="a")  # different id, same bytes
    assert ledger.verify(history) == []


def test_verify_detects_truncation():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a"), _msg(content="b")]
    ledger.extend(history)
    history.pop()
    violations = ledger.verify(history)
    assert violations and "truncated" in violations[0]


def test_verify_detects_reorder():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a"), _msg(content="b")]
    ledger.extend(history)
    history[0], history[1] = history[1], history[0]
    assert ledger.verify(history)


# --------------------------------------------------------------------------- #
# Strict vs fail-open
# --------------------------------------------------------------------------- #


def test_strict_mode_detects_in_place_mutation():
    ledger = ThreadLedger(strict=True)
    history = [_msg(content="a")]
    ledger.extend(history)
    history[0].content = "mutated in place"
    with pytest.raises(ThreadInvariantError):
        ledger.verify(history)


def test_non_strict_mode_misses_in_place_mutation_by_design():
    """Production fast path trades in-place detection for zero serialization."""
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a")]
    ledger.extend(history)
    history[0].content = "mutated in place"
    assert ledger.verify(history) == []  # id() unchanged → fast path passes


def test_non_strict_returns_violations_without_raising():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="a")]
    ledger.extend(history)
    history[0] = _msg(content="tampered")
    # Returns, does not raise
    assert ledger.verify(history)


def test_strict_resolved_from_env(monkeypatch):
    monkeypatch.setenv("NORN_STRICT_INVARIANTS", "1")
    assert ThreadLedger().strict is True
    monkeypatch.delenv("NORN_STRICT_INVARIANTS")
    assert ThreadLedger().strict is False


# --------------------------------------------------------------------------- #
# verify_derivation()
# --------------------------------------------------------------------------- #


def test_derivation_accepts_windowed_view_with_summary():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content=f"m{i}") for i in range(6)]
    ledger.extend(history)
    outbound = [
        Message(role=Role.SYSTEM, content="system prompt"),
        Message(role=Role.SYSTEM, content="## Conversation Summary\n\n- stuff"),
        *history[-2:],  # recent turns are the same objects
    ]
    assert ledger.verify_derivation(outbound, history) == []


def test_derivation_rejects_invented_message():
    ledger = ThreadLedger(strict=True)
    history = [_msg(content="a")]
    ledger.extend(history)
    outbound = [
        Message(role=Role.SYSTEM, content="system prompt"),
        _msg(content="a"),  # equal content but NOT the thread's object
    ]
    with pytest.raises(ThreadInvariantError):
        ledger.verify_derivation(outbound, history)


def test_derivation_rejects_rewritten_user_turn_non_strict():
    ledger = ThreadLedger(strict=False)
    history = [_msg(content="original")]
    ledger.extend(history)
    outbound = [_msg(content="rewritten")]
    violations = ledger.verify_derivation(outbound, history)
    assert violations and "not derived" in violations[0]


# --------------------------------------------------------------------------- #
# AgentLoop integration (runs under strict invariants via conftest autouse)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_loop_multi_round_turn_passes_invariants():
    """A tool-calling multi-round turn keeps the ledger clean end-to-end."""
    from unittest.mock import AsyncMock

    from pydantic import BaseModel

    from norn.core.agent import AgentLoop
    from norn.core.models import LLMResponse, ToolCall
    from norn.tools.base import RiskLevel, ToolContext, ToolResult
    from norn.tools.registry import ToolRegistry

    class EchoInput(BaseModel):
        text: str

    class EchoTool:
        name = "echo"
        description = "Echo"
        risk_level = RiskLevel.LOW
        input_model = EchoInput

        async def execute(self, input: EchoInput, ctx: ToolContext) -> ToolResult:
            return ToolResult(output=f"echo: {input.text}")

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})],
            ),
            LLMResponse(content="done"),
        ]
    )
    reg = ToolRegistry()
    reg.register(EchoTool())
    agent = AgentLoop(llm=llm, registry=reg, env_bootstrap=False, repo_map=False)
    response = await agent.run("go")
    assert response.content == "done"
    assert agent._ledger is not None and len(agent._ledger) > 0


@pytest.mark.asyncio
async def test_context_window_does_not_mutate_history():
    """W1.2: the sliding-window view derives; the thread itself never changes."""
    from norn.core.context import ContextManager

    manager = ContextManager(max_history_tokens=10, recent_turns_keep=1, enabled=True)
    history = [_msg(content=f"message number {i} with some padding text") for i in range(10)]
    snapshot = [m.model_copy(deep=True) for m in history]

    outbound = await manager.build_messages("system prompt", history)

    assert [m.model_dump() for m in history] == [m.model_dump() for m in snapshot]
    # Non-SYSTEM outbound messages are the thread's own objects
    ledger = ThreadLedger(strict=True)
    ledger.extend(history)
    assert ledger.verify_derivation(outbound, history) == []
