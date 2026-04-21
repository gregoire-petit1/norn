"""Tests for MCP CLI integration."""

from norn.core.config import FlagsConfig, NornConfig


def test_flags_config_has_mcp_field():
    """FlagsConfig must include the mcp flag."""
    flags = FlagsConfig()
    assert hasattr(flags, "mcp")
    assert flags.mcp is False  # default disabled


def test_norn_config_mcp_defaults():
    """NornConfig.mcp has sensible defaults."""
    cfg = NornConfig()
    assert cfg.mcp.enabled is False
    assert cfg.mcp.servers == []


def test_flags_config_mcp_can_be_enabled():
    cfg = NornConfig.model_validate({"flags": {"mcp": True}})
    assert cfg.flags.mcp is True
