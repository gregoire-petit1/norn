"""Tests for the slash command registry."""

from __future__ import annotations

import pytest
from rich.console import Console

from norn.cli.commands import CommandContext, SlashCommandRegistry, slash_command


@pytest.fixture
def registry():
    return SlashCommandRegistry()


@pytest.fixture
def ctx():
    return CommandContext(
        agent=None,
        console=Console(file=None, force_terminal=False, no_color=True, width=120),
        config=None,
        provider_factory=None,
        flag_registry=None,
    )


def test_register_and_dispatch(registry, ctx):
    """Commands registered via decorator should be dispatchable."""
    called_with = {}

    @slash_command("/ping", description="Ping test", registry=registry)
    async def cmd_ping(ctx: CommandContext, args: str) -> None:
        called_with["args"] = args

    assert "/ping" in registry.command_names()


@pytest.mark.asyncio
async def test_dispatch_calls_handler(registry, ctx):
    """dispatch() should call the registered handler."""
    result = []

    @slash_command("/echo", description="Echo args", registry=registry)
    async def cmd_echo(ctx: CommandContext, args: str) -> None:
        result.append(args)

    await registry.dispatch(ctx, "/echo hello world")
    assert result == ["hello world"]


@pytest.mark.asyncio
async def test_dispatch_unknown_command(registry, ctx):
    """Unknown command should print error, not raise."""
    await registry.dispatch(ctx, "/nonexistent foo")
    # Should not raise


def test_command_names(registry):
    """command_names() should return sorted list."""

    @slash_command("/bbb", description="B", registry=registry)
    async def cmd_b(ctx, args):
        pass

    @slash_command("/aaa", description="A", registry=registry)
    async def cmd_a(ctx, args):
        pass

    assert registry.command_names() == ["/aaa", "/bbb"]


def test_command_help_text(registry):
    """help_text() should include name and description."""

    @slash_command("/test", description="A test command", registry=registry)
    async def cmd_test(ctx, args):
        pass

    text = registry.help_text()
    assert "/test" in text
    assert "A test command" in text
