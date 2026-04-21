"""Tests for DatasetInspectorTool."""

from __future__ import annotations

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.dataset_inspector import DatasetInspectorInput, DatasetInspectorTool


@pytest.fixture
def tool():
    return DatasetInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "dataset_inspector"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_inspect_csv(tool, tmp_path):
    csv = tmp_path / "data.csv"
    csv.write_text("name,age,score\nAlice,30,90\nBob,,85\nCarol,25,70\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(csv)), ctx)
    assert result.output is not None
    assert not result.is_error
    assert "3" in result.output
    assert "name" in result.output
    assert "age" in result.output
    assert "score" in result.output


@pytest.mark.asyncio
async def test_inspect_parquet(tool, tmp_path):
    import polars as pl

    df = pl.DataFrame({"name": ["Alice", "Bob", "Carol"], "value": [1, 2, 3]})
    pq_path = tmp_path / "data.parquet"
    df.write_parquet(pq_path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(pq_path)), ctx)
    assert result.output is not None
    assert not result.is_error
    assert "3" in result.output
    assert "name" in result.output
    assert "value" in result.output


@pytest.mark.asyncio
async def test_inspect_with_sample(tool, tmp_path):
    csv = tmp_path / "data.csv"
    lines = ["x,y"] + [f"{i},{i * 10}" for i in range(5)]
    csv.write_text("\n".join(lines) + "\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        DatasetInspectorInput(file_path=str(csv), show_sample=True, sample_rows=2),
        ctx,
    )
    assert result.output is not None
    assert "Sample" in result.output


@pytest.mark.asyncio
async def test_inspect_null_stats(tool, tmp_path):
    csv = tmp_path / "nulls.csv"
    csv.write_text("a,b\n1,\n,2\n3,3\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(csv)), ctx)
    assert result.output is not None
    out_lower = result.output.lower()
    assert "null" in out_lower or "missing" in out_lower


@pytest.mark.asyncio
async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(tmp_path / "nope.csv")), ctx)
    assert result.is_error
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_inspect_unsupported_format(tool, tmp_path):
    txt = tmp_path / "data.txt"
    txt.write_text("some text")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(txt)), ctx)
    assert result.is_error
    assert "unsupported" in result.error.lower()


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_file_sets_file_not_found(tool, tmp_path):
    """Non-existent file tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolErrorType

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(tmp_path / "ghost.csv")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_unsupported_format_sets_invalid_argument(tool, tmp_path):
    """Unsupported extension tags as INVALID_ARGUMENT."""
    from norn.tools.base import ToolErrorType

    txt = tmp_path / "data.txt"
    txt.write_text("nope")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(txt)), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_corrupt_csv_sets_parse_error(tool, tmp_path):
    """Unreadable / corrupted file tags as PARSE_ERROR."""
    from norn.tools.base import ToolErrorType

    bad = tmp_path / "bad.parquet"
    bad.write_bytes(b"not a parquet file")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(bad)), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.PARSE_ERROR
