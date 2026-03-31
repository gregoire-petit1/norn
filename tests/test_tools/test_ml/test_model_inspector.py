"""Tests for ModelInspectorTool."""

from __future__ import annotations

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool


@pytest.fixture
def tool():
    return ModelInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "model_inspector"
    assert tool.risk_level == RiskLevel.LOW


async def test_inspect_model_file(tool, tmp_path):
    model_file = tmp_path / "simple_net.py"
    model_file.write_text(
        "import torch.nn as nn\n"
        "\n"
        "class SimpleNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(20, 5)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="SimpleNet"),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error
    assert "SimpleNet" in result.output
    assert "fc1" in result.output
    assert "fc2" in result.output
    assert "325" in result.output


async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(tmp_path / "nope.py"), class_name="Foo"),
        ctx,
    )
    assert result.is_error
    assert "not found" in result.error


async def test_inspect_missing_class(tool, tmp_path):
    model_file = tmp_path / "foo_module.py"
    model_file.write_text("class Foo:\n    pass\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="Bar"),
        ctx,
    )
    assert result.is_error
    assert "Bar" in result.error


async def test_inspect_with_constructor_args(tool, tmp_path):
    model_file = tmp_path / "param_net.py"
    model_file.write_text(
        "import torch.nn as nn\n"
        "\n"
        "class ParamNet(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(in_features, out_features)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(
            file_path=str(model_file),
            class_name="ParamNet",
            constructor_args={"in_features": 8, "out_features": 4},
        ),
        ctx,
    )
    assert result.output is not None
    assert not result.is_error
    assert "36" in result.output
