"""Integration tests for MCP tools."""

import sys

import pytest

from norn.mcp.adapter import MCPToolAdapter
from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport
from norn.mcp.pool import load_mcp_tools
from norn.tools.base import Tool, ToolContext
from norn.tools.registry import ToolRegistry


def _echo_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="echo",
        transport=MCPTransport.STDIO,
        command=[
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "app = FastMCP('echo')\n"
                "@app.tool()\n"
                "def echo(text: str) -> str:\n"
                "    return text\n"
                "asyncio.run(app.run_stdio_async())\n"
            ),
        ],
    )


@pytest.mark.asyncio
async def test_mcp_tools_injected_into_registry():
    """Loaded MCP adapters can be registered and queried."""
    config = MCPConfig(enabled=True, servers=[_echo_config()])
    adapters = await load_mcp_tools(config)

    registry = ToolRegistry()
    for adapter in adapters:
        registry.register(adapter)

    names = {t.name for t in registry.list_tools()}
    assert "echo" in names

    schemas = registry.get_schemas()
    schema_names = {s["name"] for s in schemas}
    assert "echo" in schema_names


@pytest.mark.asyncio
async def test_mcp_adapter_full_roundtrip(tmp_path):
    """Complete: load adapter → call tool → verify output."""
    config = MCPConfig(enabled=True, servers=[_echo_config()])
    adapters = await load_mcp_tools(config)
    echo_adapter = next((a for a in adapters if a.name == "echo"), None)
    assert echo_adapter is not None

    ctx = ToolContext(cwd=str(tmp_path))
    input_model = echo_adapter.input_model
    result = await echo_adapter.execute(input_model(text="integration test"), ctx)
    assert result.is_error is False
    assert "integration test" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_satisfies_protocol():
    """MCPToolAdapter satisfies the Tool protocol."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    adapter = MCPToolAdapter(
        server_config=_echo_config(),
        tool_name="echo",
        tool_description="Echo",
        input_schema=schema,
    )
    assert isinstance(adapter, Tool)
