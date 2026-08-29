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


# Wave 2 (B) — AXI output mode


@pytest.mark.asyncio
async def test_glob_axi_count_lead(tmp_path):
    for n in ("a.py", "b.py", "c.py"):
        (tmp_path / n).write_text("x")
    tool = GlobTool(axi_output=True)
    result = await tool.execute(
        GlobInput(pattern="*.py", path=str(tmp_path)), ToolContext(cwd=str(tmp_path))
    )
    assert result.output.splitlines()[0] == "3 files"


@pytest.mark.asyncio
async def test_glob_axi_empty_next_step(tmp_path):
    tool = GlobTool(axi_output=True)
    result = await tool.execute(
        GlobInput(pattern="*.rs", path=str(tmp_path)), ToolContext(cwd=str(tmp_path))
    )
    assert result.output.startswith("No files. Next:")


@pytest.mark.asyncio
async def test_glob_default_unchanged(tmp_path):
    tool = GlobTool()
    result = await tool.execute(
        GlobInput(pattern="*.rs", path=str(tmp_path)), ToolContext(cwd=str(tmp_path))
    )
    assert result.output == "No matches found."
