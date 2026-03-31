"""Tests for FileEditTool."""

import pytest

from norn.tools.file_edit import FileEditTool, FileEditInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return FileEditTool()


@pytest.fixture
def source_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    return 'world'\n")
    return f


def test_tool_metadata(tool):
    assert tool.name == "file_edit"
    assert tool.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_edit_replacement(tool, source_file):
    ctx = ToolContext(cwd=str(source_file.parent))
    result = await tool.execute(
        FileEditInput(
            path=str(source_file),
            old_string="return 'world'",
            new_string="return 'universe'",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "universe" in source_file.read_text()
    assert "world" not in source_file.read_text()


@pytest.mark.asyncio
async def test_edit_not_found(tool, source_file):
    ctx = ToolContext(cwd=str(source_file.parent))
    result = await tool.execute(
        FileEditInput(
            path=str(source_file),
            old_string="nonexistent string",
            new_string="replacement",
        ),
        ctx,
    )
    assert result.is_error is True
    assert "not found" in result.error.lower()
