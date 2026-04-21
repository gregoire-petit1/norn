"""Tests for ModelCardTool."""

from __future__ import annotations

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_card import ModelCardInput, ModelCardTool


@pytest.fixture
def tool():
    return ModelCardTool()


def test_tool_metadata(tool):
    assert tool.name == "model_card"
    assert tool.risk_level == RiskLevel.MEDIUM


async def test_generate_basic_card(tool, tmp_path):
    output_path = str(tmp_path / "card.md")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=output_path,
            model_name="TestNet",
            model_type="classification",
            description="A simple test model.",
        ),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error
    assert "TestNet" in result.output

    content = (tmp_path / "card.md").read_text(encoding="utf-8")
    assert "TestNet" in content
    assert "classification" in content


async def test_generate_full_card(tool, tmp_path):
    output_path = str(tmp_path / "full_card.md")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=output_path,
            model_name="FullNet",
            model_type="regression",
            description="A fully specified model.",
            version="2.1.0",
            author="Jane Doe",
            license="Apache-2.0",
            framework="PyTorch",
            datasets=["ImageNet", "COCO"],
            metrics={"accuracy": 0.9512, "f1": 0.9301, "params": 1200000},
            intended_use="Image classification in production",
            limitations=["Not suitable for video", "Requires GPU"],
            ethical_considerations="May reflect dataset biases.",
            training_details="Trained for 100 epochs on 8xA100.",
        ),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error

    content = (tmp_path / "full_card.md").read_text(encoding="utf-8")
    assert "FullNet" in content
    assert "regression" in content
    assert "2.1.0" in content
    assert "Jane Doe" in content
    assert "Apache-2.0" in content
    assert "PyTorch" in content
    assert "ImageNet" in content
    assert "COCO" in content
    assert "0.9512" in content
    assert "0.9301" in content
    assert "1200000" in content
    assert "Image classification in production" in content
    assert "Not suitable for video" in content
    assert "Requires GPU" in content
    assert "May reflect dataset biases." in content
    assert "Trained for 100 epochs on 8xA100." in content


async def test_generate_card_creates_parent_dirs(tool, tmp_path):
    output_path = str(tmp_path / "deep" / "nested" / "dir" / "card.md")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=output_path,
            model_name="DirNet",
            model_type="generative",
            description="Testing directory creation.",
        ),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error
    assert (tmp_path / "deep" / "nested" / "dir" / "card.md").exists()


async def test_card_relative_path(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path="relative_card.md",
            model_name="RelNet",
            model_type="detection",
            description="Testing relative path resolution.",
        ),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error
    assert (tmp_path / "relative_card.md").exists()
    content = (tmp_path / "relative_card.md").read_text(encoding="utf-8")
    assert "RelNet" in content


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_write_failure_sets_execution_error(tmp_path, monkeypatch):
    """Write failure tags as EXECUTION_ERROR."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_card import ModelCardInput, ModelCardTool

    def boom(self, *args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("pathlib.Path.write_text", boom)

    tool = ModelCardTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path="card.md",
            model_name="X",
            model_type="cls",
            description="d",
        ),
        ctx,
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.EXECUTION_ERROR
