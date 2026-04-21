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


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_missing_file_sets_file_not_found(tmp_path):
    """Non-existent file tags as FILE_NOT_FOUND."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool

    tool = ModelInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(tmp_path / "missing.py"), class_name="X"), ctx
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND


@pytest.mark.asyncio
async def test_missing_class_sets_invalid_argument(tmp_path):
    """Class not found tags as INVALID_ARGUMENT."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool

    pyfile = tmp_path / "m.py"
    pyfile.write_text("x = 1\n")
    tool = ModelInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(pyfile), class_name="MissingModel"), ctx
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT


@pytest.mark.asyncio
async def test_instantiation_failure_sets_execution_error(tmp_path):
    """Constructor that raises tags as EXECUTION_ERROR."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool

    pyfile = tmp_path / "broken.py"
    pyfile.write_text(
        "import torch.nn as nn\n"
        "class Broken(nn.Module):\n"
        "    def __init__(self):\n"
        "        raise RuntimeError('boom')\n"
    )
    tool = ModelInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(pyfile), class_name="Broken"), ctx
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.EXECUTION_ERROR


@pytest.mark.asyncio
async def test_not_nn_module_sets_invalid_argument(tmp_path):
    """Class that's not an nn.Module subclass tags as INVALID_ARGUMENT."""
    from norn.tools.base import ToolContext, ToolErrorType
    from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool

    pyfile = tmp_path / "plain.py"
    pyfile.write_text("class Plain:\n    pass\n")
    tool = ModelInspectorTool()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(pyfile), class_name="Plain"), ctx
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.INVALID_ARGUMENT
