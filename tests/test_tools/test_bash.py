"""Tests for BashTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.bash_tool import BashInput, BashTool


@pytest.fixture
def tool():
    return BashTool()


def test_tool_metadata(tool):
    assert tool.name == "bash"
    assert tool.risk_level == RiskLevel.HIGH


def test_default_timeout_is_300():
    """Default sized for compute-heavy steps; old 120s cut off long analyses."""
    assert BashInput(command="echo hi").timeout == 300


@pytest.mark.asyncio
async def test_timeout_reaps_process_cleanly(tool, tmp_path):
    """Timeout path must await process.wait() so the transport is reaped in the
    running loop (else __del__ fires post-close → 'Event loop is closed')."""
    ctx = ToolContext(cwd=str(tmp_path))
    # A child that ignores SIGTERM-ish quick exit; sleep is enough to force the
    # timeout branch. After it returns, no pending-subprocess warning should
    # remain — asserted indirectly by a clean completion.
    result = await tool.execute(BashInput(command="sleep 10", timeout=1), ctx)
    assert result.is_error is True
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_bash_echo(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="echo 'hello norn'"), ctx)
    assert result.is_error is False
    assert "hello norn" in result.output


@pytest.mark.asyncio
async def test_bash_failure(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="false"), ctx)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_bash_timeout(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="sleep 10", timeout=1), ctx)
    assert result.is_error is True
    assert "timeout" in result.error.lower()


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_bash_nonzero_exit_sets_execution_error(tool, tmp_path):
    """A command that exits non-zero is a runtime failure → EXECUTION_ERROR."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="false"), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.EXECUTION_ERROR.value


@pytest.mark.asyncio
async def test_bash_timeout_sets_timeout(tool, tmp_path):
    """A command exceeding the timeout window tags as TIMEOUT."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="sleep 10", timeout=1), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.TIMEOUT.value


@pytest.mark.asyncio
async def test_bash_invalid_cwd_sets_execution_error(tool, tmp_path):
    """A non-existent cwd → subprocess raises FileNotFoundError before
    timeout/exit code paths apply. Caught by the generic ``except`` →
    EXECUTION_ERROR (the catch-all bucket for unexpected runtime issues)."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path / "does_not_exist"))
    result = await tool.execute(BashInput(command="echo hi"), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.EXECUTION_ERROR.value
