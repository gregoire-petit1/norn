"""Tests for FileWriteTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.file_write import FileWriteInput, FileWriteTool


@pytest.fixture
def tool():
    return FileWriteTool()


def test_tool_metadata(tool):
    assert tool.name == "file_write"
    assert tool.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_write_new_file(tool, tmp_path):
    target = tmp_path / "new.txt"
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileWriteInput(path=str(target), content="hello world"), ctx)
    assert result.is_error is False
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_creates_parents(tool, tmp_path):
    target = tmp_path / "sub" / "dir" / "file.txt"
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileWriteInput(path=str(target), content="nested"), ctx)
    assert result.is_error is False
    assert target.read_text() == "nested"
