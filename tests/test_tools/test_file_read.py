"""Tests for FileReadTool."""

import pytest
from pathlib import Path

from norn.tools.file_read import FileReadTool, FileReadInput
from norn.tools.base import RiskLevel, ToolContext


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
