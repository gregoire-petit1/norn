"""Tests for GlobTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.glob_tool import GlobInput, GlobTool


@pytest.fixture
def tool():
    return GlobTool()


def test_tool_metadata(tool):
    assert tool.name == "glob"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_glob_pattern(tool, tmp_path):
    (tmp_path / "foo.py").touch()
    (tmp_path / "bar.py").touch()
    (tmp_path / "baz.txt").touch()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(GlobInput(pattern="*.py", path=str(tmp_path)), ctx)
    assert "foo.py" in result.output
    assert "bar.py" in result.output
    assert "baz.txt" not in result.output


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_glob_missing_directory_sets_file_not_found(tool, tmp_path):
    """A non-existent base directory tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(GlobInput(pattern="*.py", path=str(tmp_path / "ghost")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value
