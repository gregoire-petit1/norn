"""Tests for ModelEvalTool."""

from __future__ import annotations

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool


@pytest.fixture
def tool():
    return ModelEvalTool()


def test_tool_metadata(tool):
    assert tool.name == "model_eval"
    assert tool.risk_level == RiskLevel.LOW


async def test_classification_eval(tool, tmp_path):
    csv = tmp_path / "cls.csv"
    csv.write_text("y_true,y_pred\n1,1\n0,0\n1,0\n0,1\n1,1\n0,0\n1,1\n0,0\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(csv), task="classification"), ctx)

    assert result.output is not None
    assert not result.is_error
    out = result.output.lower()
    for metric in ("accuracy", "precision", "recall", "f1"):
        assert metric in out, f"Missing metric: {metric}"


async def test_regression_eval(tool, tmp_path):
    csv = tmp_path / "reg.csv"
    csv.write_text("actual,predicted\n1.0,1.1\n2.0,2.2\n3.0,2.8\n4.0,4.3\n5.0,4.9\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(
            file_path=str(csv),
            y_true_col="actual",
            y_pred_col="predicted",
            task="regression",
        ),
        ctx,
    )

    assert result.output is not None
    assert not result.is_error
    out = result.output.lower()
    assert "mse" in out
    assert "mae" in out
    assert "r2" in out or "r\u00b2" in out


async def test_auto_detect_classification(tool, tmp_path):
    csv = tmp_path / "auto.csv"
    csv.write_text("y_true,y_pred\n0,0\n1,1\n0,1\n1,0\n0,0\n1,1\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(csv)), ctx)

    assert result.output is not None
    assert not result.is_error
    assert "classification" in result.output.lower()


async def test_multiclass_eval(tool, tmp_path):
    csv = tmp_path / "multi.csv"
    csv.write_text("y_true,y_pred\n0,0\n1,1\n2,2\n0,1\n1,2\n2,0\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(csv), task="classification"), ctx)

    assert result.output is not None
    assert not result.is_error
    assert "accuracy" in result.output.lower()


async def test_missing_column(tool, tmp_path):
    csv = tmp_path / "bad_cols.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(csv)), ctx)

    assert result.is_error
    assert "y_true" in result.error


async def test_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(tmp_path / "ghost.csv")), ctx)

    assert result.is_error
    assert "not found" in result.error.lower() or "ghost.csv" in result.error


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_file_sets_file_not_found(tmp_path):
    """Non-existent CSV tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool

    tool = ModelEvalTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(tmp_path / "missing.csv")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_corrupt_csv_sets_parse_error(tmp_path):
    """Unparseable CSV tags as PARSE_ERROR."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool

    bad = tmp_path / "bad.csv"
    bad.write_bytes(b"\x00\x01\x02not,a,csv\n\xff\xfe")
    tool = ModelEvalTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(bad)), ctx)
    # Polars may either parse-fail or succeed with garbage; only check tag if errored
    if result.is_error:
        assert result.error_type == ToolErrorType.PARSE_ERROR


@pytest.mark.asyncio
async def test_missing_column_sets_invalid_argument(tmp_path):
    """Missing column tags as INVALID_ARGUMENT."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool

    csv = tmp_path / "data.csv"
    csv.write_text("a,b\n1,2\n3,4\n")
    tool = ModelEvalTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ModelEvalInput(file_path=str(csv)), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT
