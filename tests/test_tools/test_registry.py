"""Tests for tool base protocol and registry."""

import pytest
from pydantic import BaseModel

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class MockInput(BaseModel):
    query: str


class MockTool:
    name = "mock_tool"
    description = "A mock tool for testing"
    risk_level = RiskLevel.LOW
    input_model = MockInput

    async def execute(self, input: MockInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"result: {input.query}")


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def mock_tool():
    return MockTool()


def test_register_tool(registry, mock_tool):
    registry.register(mock_tool)
    assert registry.get("mock_tool") is mock_tool


def test_get_unknown_tool(registry):
    assert registry.get("nonexistent") is None


def test_list_tools(registry, mock_tool):
    registry.register(mock_tool)
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"


def test_get_schemas(registry, mock_tool):
    registry.register(mock_tool)
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "mock_tool"
    assert schema["description"] == "A mock tool for testing"
    assert "properties" in schema["parameters"]
    assert "query" in schema["parameters"]["properties"]


def test_schema_caching(registry, mock_tool):
    registry.register(mock_tool)
    schemas1 = registry.get_schemas()
    schemas2 = registry.get_schemas()
    # Same object reference = cached
    assert schemas1 is schemas2


def test_register_with_feature_flag_disabled(registry, mock_tool):
    registry.register(mock_tool, feature_flag="disabled_feature")
    # Tool registered but we can check flag status
    assert registry.get("mock_tool") is mock_tool


def test_duplicate_registration_raises(registry, mock_tool):
    registry.register(mock_tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(mock_tool)


@pytest.mark.asyncio
async def test_tool_execution(mock_tool):
    ctx = ToolContext(cwd="/tmp")
    result = await mock_tool.execute(MockInput(query="test"), ctx)
    assert result.output == "result: test"
    assert result.is_error is False


def test_flagged_tool_excluded_when_disabled():
    """Tool gated behind a disabled flag should not appear in schemas or list."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag(name="ml_tools", default=False)}
    )
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool(), feature_flag="ml_tools")

    assert registry.list_tools() == []
    assert registry.get_schemas() == []
    # But get() still works (for error messages)
    assert registry.get("mock_tool") is not None


def test_flagged_tool_included_when_enabled():
    """Tool gated behind an enabled flag should appear normally."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag(name="ml_tools", default=True)}
    )
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool(), feature_flag="ml_tools")

    assert len(registry.list_tools()) == 1
    assert len(registry.get_schemas()) == 1


def test_unflagged_tool_always_included():
    """Tool without a flag is always included."""
    flag_registry = FeatureFlagRegistry()
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool())

    assert len(registry.list_tools()) == 1


# ── W1.1 — Tool Schema Minification ──────────────────────────────────


class VerboseInput(BaseModel):
    """A verbose input model with descriptions, defaults, and examples."""

    file_path: str
    show_sample: bool = False
    sample_rows: int = 5


class VerboseTool:
    """Tool with a long description and a verbose Pydantic input model."""

    name = "verbose_tool"
    description = (
        "This is a very long description that exceeds eighty characters "
        "and should be truncated when minify is enabled on the registry."
    )
    risk_level = RiskLevel.LOW
    input_model = VerboseInput


class ShortTool:
    """Tool with a short description that stays under 80 chars."""

    name = "short_tool"
    description = "Short description."
    risk_level = RiskLevel.LOW
    input_model = MockInput


def test_get_schemas_minify_strips_defaults():
    """Minified schemas should remove 'default' from parameter properties."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=True)

    params = schemas[0]["parameters"]
    for prop in params.get("properties", {}).values():
        assert "default" not in prop


def test_get_schemas_minify_strips_title():
    """Minified schemas should remove 'title' from parameter properties."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=True)

    params = schemas[0]["parameters"]
    assert "title" not in params
    for prop in params.get("properties", {}).values():
        assert "title" not in prop


def test_get_schemas_minify_strips_additional_properties():
    """Minified schemas should remove 'additionalProperties' boilerplate."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=True)

    params = schemas[0]["parameters"]
    assert "additionalProperties" not in params


def test_get_schemas_minify_strips_examples():
    """Minified schemas should remove 'examples' from parameter properties."""
    registry = ToolRegistry()
    registry.register(VerboseTool())

    # Manually inject examples to verify removal
    schemas_full = registry.get_schemas(minify=False)
    schemas_full[0]["parameters"]["properties"]["file_path"]["examples"] = ["/tmp/test.csv"]
    # Re-invalidate cache and get minified
    registry._schema_cache = None
    registry._schema_cache_minified = None

    schemas = registry.get_schemas(minify=True)
    params = schemas[0]["parameters"]
    for prop in params.get("properties", {}).values():
        assert "examples" not in prop


def test_get_schemas_minify_truncates_long_description():
    """Tool descriptions >80 chars should be truncated with minify."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=True)

    desc = schemas[0]["description"]
    assert len(desc) <= 83  # 80 + "..."


def test_get_schemas_minify_keeps_short_description():
    """Tool descriptions <=80 chars should remain unchanged."""
    registry = ToolRegistry()
    registry.register(ShortTool())
    schemas = registry.get_schemas(minify=True)

    assert schemas[0]["description"] == "Short description."


def test_get_schemas_minify_preserves_name_and_required():
    """Minification should keep name, description, required, and property names."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=True)

    schema = schemas[0]
    assert schema["name"] == "verbose_tool"
    assert "parameters" in schema
    params = schema["parameters"]
    assert "properties" in params
    assert "file_path" in params["properties"]
    assert "show_sample" in params["properties"]
    assert "sample_rows" in params["properties"]
    # 'required' should still be present
    assert "required" in params


def test_get_schemas_minify_false_returns_full_schema():
    """minify=False should return full schema with defaults, titles, etc."""
    registry = ToolRegistry()
    registry.register(VerboseTool())
    schemas = registry.get_schemas(minify=False)

    params = schemas[0]["parameters"]
    # Full schema should have title from Pydantic
    assert "title" in params


def test_get_schemas_minify_separate_caches():
    """Minified and full schemas should use separate caches."""
    registry = ToolRegistry()
    registry.register(VerboseTool())

    full = registry.get_schemas(minify=False)
    mini = registry.get_schemas(minify=True)

    # They should not be the same object
    assert full is not mini

    # Calling again should return cached versions
    assert registry.get_schemas(minify=False) is full
    assert registry.get_schemas(minify=True) is mini


def test_get_schemas_minify_cache_invalidated_on_register():
    """Registering a new tool should invalidate both caches."""
    registry = ToolRegistry()
    registry.register(VerboseTool())

    full1 = registry.get_schemas(minify=False)
    mini1 = registry.get_schemas(minify=True)

    registry.register(ShortTool())

    full2 = registry.get_schemas(minify=False)
    mini2 = registry.get_schemas(minify=True)

    assert full1 is not full2
    assert mini1 is not mini2
    assert len(full2) == 2
    assert len(mini2) == 2


def test_get_schemas_default_is_not_minified():
    """get_schemas() without arguments should return full schemas (backward compat)."""
    registry = ToolRegistry()
    registry.register(VerboseTool())

    schemas = registry.get_schemas()
    # Should be same as minify=False
    schemas_full = registry.get_schemas(minify=False)
    assert schemas is schemas_full
