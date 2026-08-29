"""Tests for the persistent task scratchpad (wave 2 — A1)."""

import pytest

from norn.core.task_notes import TaskNotes


@pytest.fixture
def notes(tmp_path):
    return TaskNotes(tmp_path / "task_notes.md", max_chars=4000)


def test_empty_render_is_blank(notes):
    assert notes.render() == ""
    assert notes.read() == ""


def test_set_goal_and_plan_overwrite(notes):
    notes.write_section("Goal", "solve X")
    notes.write_section("Plan", "step 1\nstep 2")
    notes.write_section("Plan", "new step 1\nnew step 2")  # overwrite
    rendered = notes.render()
    assert "solve X" in rendered
    assert "new step 1" in rendered
    assert rendered.count("step 1") == 1  # old plan gone


def test_progress_and_decisions_append(notes):
    notes.write_section("Progress", "did A")
    notes.write_section("Progress", "did B")
    notes.write_section("Decisions", "chose sqlite")
    rendered = notes.render()
    assert "- did A" in rendered
    assert "- did B" in rendered
    assert "- chose sqlite" in rendered


def test_render_clamps_and_evicts_oldest_progress(tmp_path):
    n = TaskNotes(tmp_path / "n.md", max_chars=200)
    n.write_section("Goal", "important goal kept")
    n.write_section("Plan", "the plan is kept")
    for i in range(50):
        n.write_section("Progress", f"progress line number {i} with padding text")
    rendered = n.render()
    assert len(rendered) <= 200
    # Goal/Plan survive; oldest progress evicted
    assert "important goal kept" in rendered
    assert "progress line number 0 " not in rendered


def test_unknown_section_raises(notes):
    with pytest.raises(ValueError, match="Unknown section"):
        notes.write_section("Nonsense", "x")


def test_empty_content_is_noop(notes):
    notes.write_section("Goal", "g")
    notes.write_section("Goal", "   ")  # whitespace → ignored
    assert "g" in notes.render()


def test_reload_round_trip(tmp_path):
    p = tmp_path / "rt.md"
    a = TaskNotes(p)
    a.write_section("Goal", "persist me")
    a.write_section("Progress", "step done")
    # Fresh instance reads the same file
    b = TaskNotes(p)
    rendered = b.render()
    assert "persist me" in rendered
    assert "- step done" in rendered


# --------------------------------------------------------------------------- #
# AgentLoop injection (runs under strict invariants via conftest autouse)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_notes_injected_as_system_after_prefix(tmp_path):
    from unittest.mock import AsyncMock

    from norn.core.agent import AgentLoop
    from norn.core.models import LLMResponse, Role
    from norn.tools.registry import ToolRegistry

    notes = TaskNotes(tmp_path / "n.md")
    notes.write_section("Goal", "reach the summit")

    captured: dict = {}

    async def fake_complete(messages, tools=None, **kw):
        captured["messages"] = messages
        return LLMResponse(content="done")

    llm = AsyncMock()
    llm.complete = fake_complete
    agent = AgentLoop(
        llm=llm,
        registry=ToolRegistry(),
        env_bootstrap=False,
        repo_map=False,
        task_notes=notes,
    )
    await agent.run("go")

    msgs = captured["messages"]
    assert msgs[0].role == Role.SYSTEM  # base system prompt
    assert msgs[1].role == Role.SYSTEM  # injected notes
    assert "reach the summit" in (msgs[1].content or "")
    # Notes are NOT appended to the durable history (view-only injection)
    assert all("reach the summit" not in (m.content or "") for m in agent.history)


@pytest.mark.asyncio
async def test_notes_absent_when_empty(tmp_path):
    from unittest.mock import AsyncMock

    from norn.core.agent import AgentLoop
    from norn.core.models import LLMResponse, Role
    from norn.tools.registry import ToolRegistry

    notes = TaskNotes(tmp_path / "n.md")  # never written
    captured: dict = {}

    async def fake_complete(messages, tools=None, **kw):
        captured["messages"] = messages
        return LLMResponse(content="done")

    llm = AsyncMock()
    llm.complete = fake_complete
    agent = AgentLoop(
        llm=llm, registry=ToolRegistry(), env_bootstrap=False, repo_map=False, task_notes=notes
    )
    await agent.run("go")
    # Only the base system prompt, no injected note
    assert sum(1 for m in captured["messages"] if m.role == Role.SYSTEM) == 1
