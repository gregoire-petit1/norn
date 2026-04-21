"""Tests for FileEditTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.file_edit import FileEditInput, FileEditTool


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


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_edit_missing_file_sets_file_not_found(tool, tmp_path):
    """Editing a path that doesn't exist tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileEditInput(path=str(tmp_path / "ghost.txt"), old_string="a", new_string="b"),
        ctx,
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value


@pytest.mark.asyncio
async def test_edit_old_string_not_found_sets_invalid_argument(tool, source_file):
    """A non-matching ``old_string`` is a programmer/agent error: the
    arguments are ill-formed for this file. Tags as INVALID_ARGUMENT."""
    from norn.tools.base import ToolErrorType

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
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT.value


@pytest.mark.asyncio
async def test_edit_ambiguous_match_sets_invalid_argument(tool, tmp_path):
    """Multiple matches without ``replace_all=True`` is also an INVALID_ARGUMENT."""
    from norn.tools.base import ToolErrorType

    f = tmp_path / "dup.txt"
    f.write_text("foo foo foo")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileEditInput(path=str(f), old_string="foo", new_string="bar", replace_all=False),
        ctx,
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT.value
