"""Tests for MCPToolAdapter."""

import sys

import pytest

from norn.mcp.adapter import MCPToolAdapter, _make_input_model
from norn.mcp.models import MCPServerConfig, MCPTransport
from norn.tools.base import RiskLevel, Tool, ToolContext


@pytest.fixture
def echo_server_config() -> MCPServerConfig:
    """Config for a simple echo MCP server."""
    return MCPServerConfig(
        name="echo-server",
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


def test_make_input_model_has_correct_schema():
    """_make_input_model returns a class whose model_json_schema matches the given schema."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo"}},
        "required": ["text"],
    }
    Model = _make_input_model("echo", schema)
    assert Model.model_json_schema() == schema


def test_make_input_model_accepts_extra_fields():
    """Dynamic model should accept arbitrary fields (for unknown MCP tool params)."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    Model = _make_input_model("tool", schema)
    instance = Model(x=1, y="extra")  # 'y' is extra, should be accepted
    assert instance.model_dump()["x"] == 1


def test_adapter_implements_tool_protocol(echo_server_config):
    """MCPToolAdapter must satisfy the Tool protocol."""
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo text back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    assert isinstance(adapter, Tool)
    assert adapter.name == "echo"
    assert adapter.risk_level == RiskLevel.MEDIUM
    assert adapter.description == "Echo text back"


def test_adapter_schema_matches_mcp_schema(echo_server_config):
    """The adapter's input_model.model_json_schema() returns the MCP tool's schema."""
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo",
        input_schema=schema,
    )
    assert adapter.input_model.model_json_schema() == schema


@pytest.mark.asyncio
async def test_adapter_execute_stdio(echo_server_config, tmp_path):
    """MCPToolAdapter.execute() calls the MCP tool via stdio and returns its output."""
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo text",
        input_schema=schema,
    )
    ctx = ToolContext(cwd=str(tmp_path))
    InputModel = adapter.input_model
    result = await adapter.execute(InputModel(text="hello mcp"), ctx)
    assert result.is_error is False
    assert "hello mcp" in result.output


@pytest.mark.asyncio
async def test_adapter_execute_bad_server(tmp_path):
    """Adapter returns ToolResult(error=...) if server fails to start."""
    bad_config = MCPServerConfig(
        name="bad",
        transport=MCPTransport.STDIO,
        command=["python", "-c", "raise SystemExit(1)"],
    )
    schema = {"type": "object", "properties": {}}
    adapter = MCPToolAdapter(
        server_config=bad_config,
        tool_name="any_tool",
        tool_description="Should fail",
        input_schema=schema,
    )
    ctx = ToolContext(cwd=str(tmp_path))
    InputModel = adapter.input_model
    result = await adapter.execute(InputModel(), ctx)
    assert result.is_error is True
