"""MCP configuration models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class MCPTransport(StrEnum):
    """MCP server transport protocol."""

    STDIO = "stdio"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    transport: MCPTransport
    command: list[str] | None = None  # Required for STDIO transport
    url: str | None = None  # Required for SSE transport
    env: dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == MCPTransport.STDIO and not self.command:
            raise ValueError("STDIO transport requires 'command' to be set")
        if self.transport == MCPTransport.SSE and not self.url:
            raise ValueError("SSE transport requires 'url' to be set")
        return self


class MCPConfig(BaseModel):
    """Configuration for the MCP client pool."""

    enabled: bool = False
    servers: list[MCPServerConfig] = []
    timeout: float = 30.0
