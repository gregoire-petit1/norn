"""MCPToolAdapter — wraps a single MCP tool as a Norn Tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict

from norn.mcp.models import MCPServerConfig, MCPTransport
from norn.tools.base import RiskLevel, ToolContext, ToolResult


def _make_input_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model that reports the given JSON Schema via model_json_schema()."""
    _schema = schema  # capture in closure

    class _MCPInput(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
            return _schema

    _MCPInput.__name__ = f"MCPInput_{tool_name}"
    _MCPInput.__qualname__ = f"MCPInput_{tool_name}"
    return _MCPInput


class MCPToolAdapter:
    """Wraps an MCP server tool as a Norn Tool."""

    risk_level = RiskLevel.MEDIUM

    def __init__(
        self,
        server_config: MCPServerConfig,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.name = tool_name
        self.description = tool_description
        self._server_config = server_config
        self.input_model = _make_input_model(tool_name, input_schema)

    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult:
        try:
            from mcp import ClientSession
            from mcp.client.sse import sse_client
            from mcp.client.stdio import StdioServerParameters, stdio_client
        except ImportError:
            return ToolResult(
                error="mcp is not installed. Install with: uv pip install 'norn[mcp_tools]'"
            )

        # Extract arguments from input model (includes extra fields)
        arguments = {k: v for k, v in input.model_dump().items() if v is not None}

        try:
            if self._server_config.transport == MCPTransport.STDIO:
                cmd = self._server_config.command or []
                params = StdioServerParameters(
                    command=cmd[0],
                    args=cmd[1:],
                    env=self._server_config.env or None,
                )
                async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, arguments)
            else:
                url = self._server_config.url or ""
                async with sse_client(url) as (r, w), ClientSession(r, w) as session:
                    await session.initialize()
                    result = await session.call_tool(self.name, arguments)

            # Extract text content from MCP result
            texts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    texts.append(content_item.text)
                elif hasattr(content_item, "data"):
                    texts.append(f"[binary data: {len(content_item.data)} bytes]")

            return ToolResult(output="\n".join(texts) if texts else "(empty result)")

        except Exception as e:
            return ToolResult(
                error=f"MCP call failed ({self._server_config.name}/{self.name}): {e}"
            )
