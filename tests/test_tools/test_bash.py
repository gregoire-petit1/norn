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
