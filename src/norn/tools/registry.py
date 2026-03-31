"""Tool registry with schema caching."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from norn.tools.base import Tool


class ToolRegistry:
    """Registry for agent tools with cached JSON schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict[str, Any]] | None = None
        self._feature_flags: dict[str, str] = {}  # tool_name -> flag_name

    def register(self, tool: Tool, feature_flag: str | None = None) -> None:
        """Register a tool. Raises ValueError if already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        if feature_flag:
            self._feature_flags[tool.name] = feature_flag
        self._schema_cache = None  # Invalidate cache

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools. Cached for prompt efficiency."""
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
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
        scoped = ToolRegistry()
        for name in tool_names:
            tool = self.get(name)
            if tool:
                scoped.register(tool)
        return scoped
