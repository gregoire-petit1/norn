"""Tests for TensorInspectorTool."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool


@pytest.fixture
def tool():
    return TensorInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "tensor_inspector"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_inspect_state_dict(tool, tmp_path):
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(path)), ctx)

    assert result.output is not None
    assert not result.is_error
    assert "0.weight" in result.output
    assert "0.bias" in result.output
    assert "2.weight" in result.output
    # Linear(10, 20) weight shape is [20, 10]
    assert "[20, 10]" in result.output


@pytest.mark.asyncio
async def test_inspect_single_tensor(tool, tmp_path):
    tensor = torch.randn(3, 4, 5)
    path = tmp_path / "tensor.pt"
    torch.save(tensor, path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(path)), ctx)

    assert result.output is not None
    assert not result.is_error
    assert "[3, 4, 5]" in result.output


@pytest.mark.asyncio
async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        TensorInspectorInput(file_path=str(tmp_path / "nonexistent.pt")), ctx
    )

    assert result.is_error
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_inspect_dict_checkpoint(tool, tmp_path):
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters())
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": 42,
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(path)), ctx)

    assert result.output is not None
    assert not result.is_error
    assert "model_state_dict" in result.output


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_file_sets_file_not_found(tmp_path):
    """Non-existent file tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool

    tool = TensorInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(tmp_path / "ghost.pt")), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_corrupt_file_sets_parse_error(tmp_path):
    """Unloadable file tags as PARSE_ERROR."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool

    bad = tmp_path / "bad.pt"
    bad.write_bytes(b"not a torch file")
    tool = TensorInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(bad)), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.PARSE_ERROR
