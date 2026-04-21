"""Tests for GrepTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.grep_tool import GrepInput, GrepTool


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


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_grep_missing_path_sets_file_not_found(tool, tmp_path):
    """Path that doesn't exist tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(GrepInput(pattern="x", path=str(tmp_path / "ghost")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value


@pytest.mark.asyncio
async def test_grep_invalid_regex_sets_invalid_argument(tool, tmp_path):
    """A malformed regex pattern is an INVALID_ARGUMENT (caller's responsibility)."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    # Unbalanced bracket → re.error
    result = await tool.execute(GrepInput(pattern="[unclosed", path=str(tmp_path)), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT.value
