"""Tests for GrepTool."""

import pytest

from norn.tools.grep_tool import GrepTool, GrepInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return GrepTool()


def test_tool_metadata(tool):
    assert tool.name == "grep"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_grep_pattern(tool, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(GrepInput(pattern="def \\w+", path=str(tmp_path)), ctx)
    assert "hello" in result.output
    assert "goodbye" in result.output


@pytest.mark.asyncio
async def test_grep_no_matches(tool, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(GrepInput(pattern="class Foo", path=str(tmp_path)), ctx)
    assert result.output is not None  # Empty but no error
