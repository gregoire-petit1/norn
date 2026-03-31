"""Tool registry with schema caching and feature flag support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from norn.flags.registry import FeatureFlagRegistry
    from norn.tools.base import Tool


class ToolRegistry:
    """Registry for agent tools with cached JSON schemas."""

    def __init__(self, flag_registry: FeatureFlagRegistry | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict[str, Any]] | None = None
        self._feature_flags: dict[str, str] = {}  # tool_name -> flag_name
        self._flag_registry: FeatureFlagRegistry | None = flag_registry

    def register(self, tool: Tool, feature_flag: str | None = None) -> None:
        """Register a tool. Raises ValueError if already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        if feature_flag:
            self._feature_flags[tool.name] = feature_flag
        self._schema_cache = None  # Invalidate cache

    def _is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled via its feature flag."""
        flag_name = self._feature_flags.get(name)
        if flag_name is None:
            return True  # No flag = always enabled
        if self._flag_registry is None:
            return True  # No flag registry = always enabled
        return self._flag_registry.is_enabled(flag_name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name (regardless of flag status)."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all enabled tools."""
        return [t for t in self._tools.values() if self._is_tool_enabled(t.name)]

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for enabled tools. Cached for prompt efficiency."""
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
            if not self._is_tool_enabled(tool.name):
                continue
            schema = tool.input_model.model_json_schema()
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                }
            )
        self._schema_cache = schemas
        return self._schema_cache

    def scoped(self, tool_names: list[str]) -> ToolRegistry:
        """Create a new registry with only the specified tools."""
        scoped = ToolRegistry(flag_registry=self._flag_registry)
        for name in tool_names:
            tool = self.get(name)
            if tool:
                flag = self._feature_flags.get(name)
                scoped.register(tool, feature_flag=flag)
        return scoped
