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
