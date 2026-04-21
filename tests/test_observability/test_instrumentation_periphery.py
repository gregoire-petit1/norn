"""Integration tests: tool execution and permission checker emit events.

Verifies Task 6 instrumentation:

- ``AgentLoop._execute_tool`` emits ``tool.call`` with ``tool_name``,
  ``duration_ms``, ``success``, and harmonised ``error_type`` /
  ``error_message`` on failure. The helper :func:`measure_and_log`
  cannot be used here: tool errors are captured into ``ToolResult(error=...)``
  rather than raised, so the non-raise path would always report
  ``success=True``. A direct emission is used instead (mirrors the
  one-shot ``routing.decision`` / ``fallback`` pattern from T5).
- ``PermissionChecker.check`` emits ``permission.decision`` with
  ``tool_name``, ``mode``, ``granted`` and ``reason``. The event field
  is named ``granted`` per the observability design doc, even though
  the Python attribute on :class:`PermissionDecision` is ``approved``.

All emissions are wrapped in ``contextlib.suppress(Exception)`` so a
logging failure can never affect caller control flow (fail-open).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from norn.core.config import LoggingConfig, PermissionMode
from norn.core.models import ToolCall
from norn.observability.logger import init_logging, new_session
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.permissions.models import PermissionRequest
from norn.tools.base import RiskLevel, ToolResult

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
    """File-only logging into tmp_path + fresh session id."""
    cfg = LoggingConfig(
        enabled=True,
        output="file",
        file_dir=str(tmp_path),
        include_cost=False,
    )
    init_logging(cfg)
    new_session()
    yield tmp_path
    import logging as _logging

    for h in list(_logging.getLogger().handlers):
        h.close()
        _logging.getLogger().removeHandler(h)


# --------------------------------------------------------------------------- #
# Fake tool + AgentLoop helper
# --------------------------------------------------------------------------- #


class _EchoInput(BaseModel):
    msg: str = "hi"


def _make_tool(
    *,
    name: str = "echo",
    execute_side_effect: Exception | ToolResult | None = None,
) -> MagicMock:
    """Build a protocol-compatible mock Tool."""
    tool = MagicMock()
    tool.name = name
    tool.description = "echo tool"
    tool.risk_level = RiskLevel.LOW
    tool.input_model = _EchoInput
    if execute_side_effect is None:
        tool.execute = AsyncMock(return_value=ToolResult(output="ok"))
    elif isinstance(execute_side_effect, Exception):
        tool.execute = AsyncMock(side_effect=execute_side_effect)
    else:
        tool.execute = AsyncMock(return_value=execute_side_effect)
    return tool


def _make_agent(tool: MagicMock):
    from norn.core.agent import AgentLoop
    from norn.tools.registry import ToolRegistry

    registry = ToolRegistry()
    registry.register(tool)
    llm = MagicMock()
    return AgentLoop(llm=llm, registry=registry)


# --------------------------------------------------------------------------- #
# tool.call
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_execute_tool_emits_tool_call_on_success(log_dir: Path) -> None:
    tool = _make_tool()
    agent = _make_agent(tool)

    result = await agent._execute_tool(ToolCall(id="c1", name="echo", arguments={"msg": "x"}))
    assert result.error is None

    events = _events(_today_file(log_dir))
    tool_calls = [e for e in events if e.get("event") == "tool.call"]
    assert len(tool_calls) == 1
    ev = tool_calls[0]
    assert ev["tool_name"] == "echo"
    assert ev["success"] is True
    assert isinstance(ev["duration_ms"], int)
    assert ev["duration_ms"] >= 0
    assert "error_type" not in ev
    assert "error_message" not in ev


@pytest.mark.asyncio
async def test_agent_execute_tool_emits_tool_call_on_failure(log_dir: Path) -> None:
    tool = _make_tool(execute_side_effect=RuntimeError("kaboom"))
    agent = _make_agent(tool)

    # Existing behaviour: exceptions are caught and wrapped in ToolResult.
    result = await agent._execute_tool(ToolCall(id="c1", name="echo", arguments={"msg": "x"}))
    assert result.error is not None
    assert "kaboom" in result.error

    events = _events(_today_file(log_dir))
    tool_calls = [e for e in events if e.get("event") == "tool.call"]
    assert len(tool_calls) == 1
    ev = tool_calls[0]
    assert ev["tool_name"] == "echo"
    assert ev["success"] is False
    assert ev["error_type"] == "ToolExecutionError"
    assert "kaboom" in ev["error_message"]
    # Harmonised schema: no conflated `error` field.
    assert "error" not in ev


# --------------------------------------------------------------------------- #
# permission.decision
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_permission_checker_emits_decision_on_grant(log_dir: Path) -> None:
    checker = PermissionChecker(mode=PermissionMode.YOLO, classifier=RiskClassifier())
    req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "/tmp/x"})

    decision = await checker.check(req)
    assert decision.approved is True

    events = _events(_today_file(log_dir))
    decisions = [e for e in events if e.get("event") == "permission.decision"]
    assert len(decisions) == 1
    ev = decisions[0]
    assert ev["tool_name"] == "file_read"
    assert ev["granted"] is True
    assert ev["mode"] == "yolo"


@pytest.mark.asyncio
async def test_permission_checker_emits_decision_on_deny(log_dir: Path) -> None:
    # STRICT + destructive bash command => denied with a reason, no prompt.
    checker = PermissionChecker(mode=PermissionMode.STRICT, classifier=RiskClassifier())
    req = PermissionRequest(
        tool_name="bash",
        risk_level="high",
        arguments={"command": "rm -rf /"},
    )

    decision = await checker.check(req)
    assert decision.approved is False
    assert decision.reason is not None

    events = _events(_today_file(log_dir))
    decisions = [e for e in events if e.get("event") == "permission.decision"]
    assert len(decisions) == 1
    ev = decisions[0]
    assert ev["tool_name"] == "bash"
    assert ev["granted"] is False
    assert ev["mode"] == "strict"
    assert ev["reason"]  # non-empty reason when denied


# --------------------------------------------------------------------------- #
# session_id propagation
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_session_id_shared_across_periphery(log_dir: Path) -> None:
    sid = new_session()

    tool = _make_tool()
    agent = _make_agent(tool)
    await agent._execute_tool(ToolCall(id="c1", name="echo", arguments={"msg": "x"}))

    checker = PermissionChecker(mode=PermissionMode.YOLO, classifier=RiskClassifier())
    await checker.check(
        PermissionRequest(tool_name="echo", risk_level="low", arguments={"msg": "x"}),
    )

    events = _events(_today_file(log_dir))
    tool_calls = [e for e in events if e.get("event") == "tool.call"]
    decisions = [e for e in events if e.get("event") == "permission.decision"]
    assert len(tool_calls) == 1
    assert len(decisions) == 1
    assert tool_calls[0]["session_id"] == sid
    assert decisions[0]["session_id"] == sid
