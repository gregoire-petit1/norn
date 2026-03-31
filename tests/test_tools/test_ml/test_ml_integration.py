"""Integration tests for ML tools with agent infrastructure."""

import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.base import ToolContext
from norn.tools.ml.dataset_inspector import DatasetInspectorInput, DatasetInspectorTool
from norn.tools.ml.model_card import ModelCardInput, ModelCardTool
from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool
from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool
from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool
from norn.tools.registry import ToolRegistry


@pytest.fixture
def full_ml_registry():
    """Build a full registry with ML tools enabled."""
    flags = FeatureFlagRegistry(flags={"ml_tools": FeatureFlag("ml_tools", True, "ML tools")})
    flags.apply_config({"ml_tools": True})
    registry = ToolRegistry(flag_registry=flags)
    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")
    return registry


def test_all_ml_tools_have_schemas(full_ml_registry):
    """All ML tools should produce valid JSON schemas."""
    schemas = full_ml_registry.get_schemas()
    names = {s["name"] for s in schemas}
    assert names == {
        "model_inspector",
        "tensor_inspector",
        "dataset_inspector",
        "model_eval",
        "model_card",
    }
    for schema in schemas:
        assert "parameters" in schema
        assert "description" in schema


def test_ml_tools_retrieved_by_name(full_ml_registry):
    """Each ML tool can be retrieved by name."""
    for name in [
        "model_inspector",
        "tensor_inspector",
        "dataset_inspector",
        "model_eval",
        "model_card",
    ]:
        tool = full_ml_registry.get(name)
        assert tool is not None
        assert tool.name == name


@pytest.mark.asyncio
async def test_ml_pipeline_inspect_and_evaluate(tmp_path):
    """Simulate: inspect model -> generate predictions -> evaluate -> write card."""
    ctx = ToolContext(cwd=str(tmp_path))

    # Step 1: Inspect model
    model_file = tmp_path / "net.py"
    model_file.write_text(
        "import torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    inspector = ModelInspectorTool()
    result = await inspector.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="Net"), ctx
    )
    assert result.is_error is False
    assert "Net" in result.output

    # Step 2: Create predictions CSV and evaluate
    preds = tmp_path / "preds.csv"
    preds.write_text("y_true,y_pred\n0,0\n1,1\n0,1\n1,1\n0,0\n")
    evaluator = ModelEvalTool()
    result = await evaluator.execute(
        ModelEvalInput(file_path=str(preds), y_true_col="y_true", y_pred_col="y_pred"), ctx
    )
    assert result.is_error is False
    assert "accuracy" in result.output.lower()

    # Step 3: Generate model card
    card_tool = ModelCardTool()
    result = await card_tool.execute(
        ModelCardInput(
            output_path=str(tmp_path / "MODEL_CARD.md"),
            model_name="Net",
            model_type="classification",
            description="Simple 2-class classifier.",
            metrics={"accuracy": 0.8, "f1": 0.75},
        ),
        ctx,
    )
    assert result.is_error is False
    assert (tmp_path / "MODEL_CARD.md").exists()


@pytest.mark.asyncio
async def test_dataset_then_tensor_inspection(tmp_path):
    """Inspect a dataset and a checkpoint in sequence."""
    import torch
    import torch.nn as nn

    ctx = ToolContext(cwd=str(tmp_path))

    # Create a dataset
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("feature_1,feature_2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
    ds_tool = DatasetInspectorTool()
    result = await ds_tool.execute(DatasetInspectorInput(file_path=str(csv_file)), ctx)
    assert result.is_error is False
    assert "3" in result.output  # 3 rows

    # Save and inspect a checkpoint
    model = nn.Linear(2, 1)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    tensor_tool = TensorInspectorTool()
    result = await tensor_tool.execute(TensorInspectorInput(file_path=str(ckpt_path)), ctx)
    assert result.is_error is False
    assert "weight" in result.output
