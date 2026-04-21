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


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_write_to_unwritable_path_sets_execution_error(tool, tmp_path):
    """Any I/O failure on write (permission, ENOSPC, …) tags as EXECUTION_ERROR.

    We provoke it by trying to write inside a regular file (treated as a dir
    by ``parent.mkdir`` → NotADirectoryError). The exact OSError subclass is
    OS-dependent, so we only assert the taxonomy bucket.
    """
    from norn.tools.base import ToolErrorType

    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    target = blocker / "child.txt"  # parent path is a regular file
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(FileWriteInput(path=str(target), content="x"), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.EXECUTION_ERROR.value
