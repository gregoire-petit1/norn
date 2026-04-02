"""Tests for MCP pool / load_mcp_tools."""

import sys

import pytest

from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport
from norn.mcp.pool import load_mcp_tools


def _echo_server_config(name: str = "echo") -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
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
                "@app.tool()\n"
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n"
                "asyncio.run(app.run_stdio_async())\n"
            ),
        ],
    )


@pytest.mark.asyncio
async def test_load_tools_from_one_server():
    """Returns adapters for all tools in a server."""
    config = MCPConfig(enabled=True, servers=[_echo_server_config()])
    adapters = await load_mcp_tools(config)
    names = {a.name for a in adapters}
    assert "echo" in names
    assert "add" in names


@pytest.mark.asyncio
async def test_load_tools_from_multiple_servers():
    """Returns tools from all servers combined."""
    config = MCPConfig(
        enabled=True,
        servers=[_echo_server_config("s1"), _echo_server_config("s2")],
    )
    adapters = await load_mcp_tools(config)
    # Both servers expose echo + add = 4 adapters total
    assert len(adapters) == 4


@pytest.mark.asyncio
async def test_load_tools_bad_server_doesnt_crash():
    """A failing server is skipped; other servers still load."""
    bad = MCPServerConfig(
        name="bad",
        transport=MCPTransport.STDIO,
        command=["python", "-c", "raise SystemExit(1)"],
    )
    config = MCPConfig(enabled=True, servers=[bad, _echo_server_config("good")])
    adapters = await load_mcp_tools(config)
    # bad server fails silently; good server loads
    names = {a.name for a in adapters}
    assert "echo" in names


@pytest.mark.asyncio
async def test_load_tools_empty_config():
    """Empty server list returns empty adapter list."""
    config = MCPConfig(enabled=True, servers=[])
    adapters = await load_mcp_tools(config)
    assert adapters == []


@pytest.mark.asyncio
async def test_load_tools_disabled_config():
    """Disabled MCPConfig returns empty list without connecting."""
    config = MCPConfig(enabled=False, servers=[_echo_server_config()])
    adapters = await load_mcp_tools(config)
    assert adapters == []
