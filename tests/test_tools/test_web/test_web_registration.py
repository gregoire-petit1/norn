"""Tests for web tool registration."""

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.registry import ToolRegistry
from norn.tools.web.web_fetch import WebFetchTool
from norn.tools.web.web_search import WebSearchTool


def _build_web_registry(enabled: bool) -> ToolRegistry:
    flag_registry = FeatureFlagRegistry(
        flags={"web_search": FeatureFlag("web_search", enabled, "Web tools")}
    )
    flag_registry.apply_config({"web_search": enabled})
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(WebFetchTool(), feature_flag="web_search")
    registry.register(WebSearchTool(), feature_flag="web_search")
    return registry


def test_web_tools_enabled():
    registry = _build_web_registry(enabled=True)
    names = {t.name for t in registry.list_tools()}
    assert "web_fetch" in names
    assert "web_search" in names


def test_web_tools_disabled():
    registry = _build_web_registry(enabled=False)
    names = {t.name for t in registry.list_tools()}
    assert "web_fetch" not in names
    assert "web_search" not in names


def test_web_tools_schemas_when_enabled():
    registry = _build_web_registry(enabled=True)
    schema_names = {s["name"] for s in registry.get_schemas()}
    assert "web_fetch" in schema_names
    assert "web_search" in schema_names
