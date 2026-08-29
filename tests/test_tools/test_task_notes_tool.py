"""Tests for the task_notes tool (wave 2 — A1)."""

import pytest

from norn.core.task_notes import TaskNotes
from norn.tools.base import ToolContext
from norn.tools.task_notes_tool import TaskNotesInput, TaskNotesTool


@pytest.fixture
def tool(tmp_path):
    return TaskNotesTool(TaskNotes(tmp_path / "n.md"))


@pytest.mark.asyncio
async def test_set_plan_then_read(tool):
    ctx = ToolContext(cwd=".")
    r = await tool.execute(TaskNotesInput(action="set_plan", content="1. a\n2. b"), ctx)
    assert not r.is_error and "Plan" in r.output
    r = await tool.execute(TaskNotesInput(action="read"), ctx)
    assert "1. a" in r.output


@pytest.mark.asyncio
async def test_note_progress_appends(tool):
    ctx = ToolContext(cwd=".")
    await tool.execute(TaskNotesInput(action="note_progress", content="wrote parser"), ctx)
    await tool.execute(TaskNotesInput(action="note_progress", content="added tests"), ctx)
    r = await tool.execute(TaskNotesInput(action="read"), ctx)
    assert "wrote parser" in r.output and "added tests" in r.output


@pytest.mark.asyncio
async def test_read_empty(tool):
    r = await tool.execute(TaskNotesInput(action="read"), ctx=ToolContext(cwd="."))
    assert "empty" in r.output.lower()


def test_tool_registers_and_gated(tmp_path):
    from norn.cli.main import _build_flag_registry, _build_registry, _build_task_notes
    from norn.core.config import NornConfig

    cfg = NornConfig()
    cfg.agent.task_notes = True
    tn = _build_task_notes(cfg)
    assert tn is not None
    reg = _build_registry(_build_flag_registry(cfg), config=cfg, task_notes=tn)
    assert "task_notes" in [t.name for t in reg.list_tools()]

    cfg2 = NornConfig()
    cfg2.agent.task_notes = False
    assert _build_task_notes(cfg2) is None
    reg2 = _build_registry(_build_flag_registry(cfg2), config=cfg2, task_notes=None)
    assert "task_notes" not in [t.name for t in reg2.list_tools()]
