"""Tests for batched bash commands (wave 3 — one LLM round-trip, N commands).

Motivation: paired tb2 run showed Norn issuing 1.14 tool calls per LLM call
vs 3.39 for Terminus-2 for the same tool work — 2.5x the round-trips. The
prompt alone did not fix it (the model correctly serialises dependent
steps); making the tool accept a list makes batching structural.
"""

import pytest

from norn.tools.base import ToolContext
from norn.tools.bash_tool import BashInput, BashTool


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(cwd=str(tmp_path))


@pytest.mark.asyncio
async def test_batch_runs_all_and_labels_each(ctx):
    r = await BashTool().execute(BashInput(commands=["echo one", "echo two"]), ctx)
    assert not r.is_error
    assert "$ echo one  [ok]\none" in r.output
    assert "$ echo two  [ok]\ntwo" in r.output


@pytest.mark.asyncio
async def test_batch_stops_at_first_failure(ctx):
    r = await BashTool().execute(BashInput(commands=["echo start", "false", "echo NEVER"]), ctx)
    assert r.is_error
    assert r.error.startswith("Exit code 1")
    assert "[rc=1]" in r.error
    assert "NEVER" not in r.error
    assert "1 command(s) not run" in r.error


@pytest.mark.asyncio
async def test_batch_shares_one_shell(ctx):
    """State carries across commands: a cd or a variable set earlier persists."""
    r = await BashTool().execute(
        BashInput(commands=["X=hello", "echo $X", "mkdir sub && cd sub", "pwd"]), ctx
    )
    assert not r.is_error
    assert "hello" in r.output
    assert r.output.rstrip().endswith("sub")


@pytest.mark.asyncio
async def test_batch_stderr_appended_once(ctx):
    r = await BashTool().execute(BashInput(commands=["echo out", "echo err >&2"]), ctx)
    assert not r.is_error
    assert r.output.count("STDERR:") == 1
    assert "err" in r.output


@pytest.mark.asyncio
async def test_single_command_path_unchanged(ctx):
    """Legacy `command` keeps its exact output shape (no batch labels)."""
    r = await BashTool().execute(BashInput(command="echo legacy"), ctx)
    assert r.output.strip() == "legacy"
    assert "[ok]" not in r.output and "$ " not in r.output


@pytest.mark.asyncio
async def test_commands_wins_over_command(ctx):
    r = await BashTool().execute(BashInput(command="echo IGNORED", commands=["echo batch"]), ctx)
    assert "batch" in r.output and "IGNORED" not in r.output


@pytest.mark.asyncio
async def test_empty_input_rejected(ctx):
    r = await BashTool().execute(BashInput(), ctx)
    assert r.error_type == "InvalidArgument"
    r2 = await BashTool().execute(BashInput(commands=["", "  "]), ctx)
    assert r2.error_type == "InvalidArgument"


def test_split_batch_handles_unrun_commands():
    out = "a\n__SEP__0:0\nb\n__SEP__1:2\n"
    rendered = BashTool._split_batch(out, ["echo a", "false", "echo c"], "__SEP__")
    assert "$ echo a  [ok]\na" in rendered
    assert "$ false  [rc=2]\nb" in rendered
    assert "1 command(s) not run" in rendered
