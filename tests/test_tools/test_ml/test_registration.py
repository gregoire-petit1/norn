"""Tests for ML tool registration."""

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.ml.dataset_inspector import DatasetInspectorTool
from norn.tools.ml.model_card import ModelCardTool
from norn.tools.ml.model_eval import ModelEvalTool
from norn.tools.ml.model_inspector import ModelInspectorTool
from norn.tools.ml.tensor_inspector import TensorInspectorTool
from norn.tools.registry import ToolRegistry


def _build_ml_registry(ml_enabled: bool) -> ToolRegistry:
    """Build a registry with ML tools for testing."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag("ml_tools", ml_enabled, "ML tools")}
    )
    flag_registry.apply_config({"ml_tools": ml_enabled})

    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")
    return registry


def test_ml_tools_enabled():
    """When ml_tools flag is on, all 5 ML tools are listed."""
    registry = _build_ml_registry(ml_enabled=True)
    names = {t.name for t in registry.list_tools()}
    assert "model_inspector" in names
    assert "tensor_inspector" in names
    assert "dataset_inspector" in names
    assert "model_eval" in names
    assert "model_card" in names


def test_ml_tools_disabled():
    """When ml_tools flag is off, no ML tools are listed."""
    registry = _build_ml_registry(ml_enabled=False)
    names = {t.name for t in registry.list_tools()}
    assert "model_inspector" not in names
    assert "tensor_inspector" not in names
    assert "dataset_inspector" not in names
    assert "model_eval" not in names
    assert "model_card" not in names


def test_ml_tools_schemas_respect_flag():
    """Schemas should only include enabled tools."""
    registry_on = _build_ml_registry(ml_enabled=True)
    registry_off = _build_ml_registry(ml_enabled=False)

    schemas_on = registry_on.get_schemas()
    schemas_off = registry_off.get_schemas()

    ml_names_on = {s["name"] for s in schemas_on}
    ml_names_off = {s["name"] for s in schemas_off}

    assert "model_inspector" in ml_names_on
    assert "model_inspector" not in ml_names_off
