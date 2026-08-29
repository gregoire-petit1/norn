"""Tests for FileReadTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.file_read import FileReadInput, FileReadTool


@pytest.fixture
def tool():
    return FileReadTool()


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n")
    return f


def test_tool_metadata(tool):
    assert tool.name == "file_read"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_read_file(tool, tmp_file):
    ctx = ToolContext(cwd=str(tmp_file.parent))
    result = await tool.execute(FileReadInput(path=str(tmp_file)), ctx)
    assert "line1" in result.output
    assert "line2" in result.output
    assert result.is_error is False


@pytest.mark.asyncio
async def test_read_nonexistent(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileReadInput(path=str(tmp_path / "nope.txt")), ctx)
    assert result.is_error is True
    assert "not found" in result.error.lower() or "No such file" in result.error


@pytest.mark.asyncio
async def test_read_directory(tool, tmp_path):
    (tmp_path / "a.py").touch()
    (tmp_path / "b.py").touch()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileReadInput(path=str(tmp_path)), ctx)
    assert "a.py" in result.output
    assert "b.py" in result.output


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_read_missing_file_sets_file_not_found(tool, tmp_path):
    """Path that doesn't exist must tag the failure as FILE_NOT_FOUND."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileReadInput(path=str(tmp_path / "nope.txt")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value


# Wave 2 (B) — AXI output mode


@pytest.mark.asyncio
async def test_file_read_axi_line_header(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("\n".join(f"line {i}" for i in range(100)))
    tool = FileReadTool(axi_output=True)
    result = await tool.execute(
        FileReadInput(path=str(f), offset=0, limit=10), ToolContext(cwd=str(tmp_path))
    )
    assert result.output.splitlines()[0] == f"{f}: lines 1-10 of 100"


@pytest.mark.asyncio
async def test_file_read_axi_dir_header(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "a.txt").write_text("x")
    tool = FileReadTool(axi_output=True)
    result = await tool.execute(
        FileReadInput(path=str(tmp_path)), ToolContext(cwd=str(tmp_path))
    )
    assert result.output.splitlines()[0] == "2 entries (1 dirs, 1 files)"


@pytest.mark.asyncio
async def test_file_read_default_unchanged(tmp_path):
    f = tmp_path / "x.py"
    f.write_text("a\nb\nc")
    tool = FileReadTool()
    result = await tool.execute(FileReadInput(path=str(f)), ToolContext(cwd=str(tmp_path)))
    assert result.output == "1: a\n2: b\n3: c"
