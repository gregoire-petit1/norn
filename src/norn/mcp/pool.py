"""MCP pool — loads tool adapters from all configured MCP servers."""

from __future__ import annotations

import logging

from norn.mcp.adapter import MCPToolAdapter
from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport

logger = logging.getLogger(__name__)


async def load_mcp_tools(config: MCPConfig) -> list[MCPToolAdapter]:
    """Connect to all MCP servers and return a flat list of MCPToolAdapter instances.

    Servers that fail to connect or list tools are skipped with a warning.
    """
    if not config.enabled:
        return []

    try:
        from mcp import ClientSession  # noqa: F401 — verify import works
        from mcp.client.stdio import stdio_client  # noqa: F401
    except ImportError:
        logger.warning("mcp package not installed — MCP tools unavailable")
        return []

    adapters: list[MCPToolAdapter] = []

    for server in config.servers:
        try:
            tools = await _list_tools(server, config.timeout)
            for mcp_tool in tools:
                adapter = MCPToolAdapter(
                    server_config=server,
                    tool_name=mcp_tool.name,
                    tool_description=mcp_tool.description or mcp_tool.name,
                    input_schema=mcp_tool.inputSchema,
                )
                adapters.append(adapter)
        except Exception as exc:
            logger.warning("Failed to load tools from MCP server '%s': %s", server.name, exc)

    return adapters


async def _list_tools(server: MCPServerConfig, timeout: float) -> list:
    """Connect to a server, list its tools, then disconnect."""
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    if server.transport == MCPTransport.STDIO:
        cmd = server.command or []
        params = StdioServerParameters(
            command=cmd[0],
            args=cmd[1:],
            env=server.env or None,
        )
        async with stdio_client(params) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools
    else:
        url = server.url or ""
        async with sse_client(url) as (r, w), ClientSession(r, w) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools
