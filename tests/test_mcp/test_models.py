"""Tests for MCP config models."""

import pytest

from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport


def test_mcp_transport_values():
    assert MCPTransport.STDIO == "stdio"
    assert MCPTransport.SSE == "sse"


def test_server_config_stdio():
    cfg = MCPServerConfig(
        name="filesystem",
        transport=MCPTransport.STDIO,
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    assert cfg.name == "filesystem"
    assert cfg.transport == MCPTransport.STDIO
    assert cfg.command == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert cfg.url is None
    assert cfg.env == {}


def test_server_config_sse():
    cfg = MCPServerConfig(
        name="remote",
        transport=MCPTransport.SSE,
        url="http://localhost:8000/sse",
    )
    assert cfg.url == "http://localhost:8000/sse"
    assert cfg.command is None


def test_server_config_validation_stdio_needs_command():
    """STDIO transport without command should fail validation."""
    with pytest.raises(ValueError, match="command"):
        MCPServerConfig(name="bad", transport=MCPTransport.STDIO)


def test_server_config_validation_sse_needs_url():
    """SSE transport without url should fail validation."""
    with pytest.raises(ValueError, match="url"):
        MCPServerConfig(name="bad", transport=MCPTransport.SSE)


def test_mcp_config_defaults():
    cfg = MCPConfig()
    assert cfg.servers == []
    assert cfg.enabled is False
    assert cfg.timeout > 0


def test_mcp_config_with_servers():
    cfg = MCPConfig(
        enabled=True,
        servers=[
            MCPServerConfig(
                name="fs",
                transport=MCPTransport.STDIO,
                command=["python", "-m", "myserver"],
            )
        ],
    )
    assert cfg.enabled is True
    assert len(cfg.servers) == 1
    assert cfg.servers[0].name == "fs"


def test_norn_config_has_mcp():
    """NornConfig should include mcp sub-config."""
    from norn.core.config import NornConfig

    cfg = NornConfig()
    assert hasattr(cfg, "mcp")
    assert isinstance(cfg.mcp, MCPConfig)
