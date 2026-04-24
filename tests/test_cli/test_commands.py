"""Tests for the slash command registry."""

from __future__ import annotations

import pytest
from rich.console import Console

from unittest.mock import MagicMock

from norn.cli.commands import (
    CommandContext,
    SlashCommandRegistry,
    _ExitRequested,
    build_default_registry,
    slash_command,
)


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


# --- Session commands tests ---


@pytest.fixture
def default_registry():
    return build_default_registry()


@pytest.fixture
def full_ctx():
    agent = MagicMock()
    agent.history = [MagicMock(), MagicMock()]
    agent.user_message_count = 1
    agent.tool_call_count = 0
    return CommandContext(
        agent=agent,
        console=Console(file=None, force_terminal=False, no_color=True, width=120),
        config=None,
        provider_factory=None,
        flag_registry=None,
    )


def test_default_registry_has_help(default_registry):
    assert "/help" in default_registry.command_names()


def test_default_registry_has_exit(default_registry):
    assert "/exit" in default_registry.command_names()


def test_default_registry_has_clear(default_registry):
    assert "/clear" in default_registry.command_names()


@pytest.mark.asyncio
async def test_clear_resets_history(default_registry, full_ctx):
    await default_registry.dispatch(full_ctx, "/clear")
    assert full_ctx.agent.history == []


@pytest.mark.asyncio
async def test_exit_raises(default_registry, full_ctx):
    with pytest.raises(_ExitRequested):
        await default_registry.dispatch(full_ctx, "/exit")


@pytest.mark.asyncio
async def test_model_no_args_shows_current(default_registry, full_ctx):
    """``/model`` without args should show the current model."""
    provider = MagicMock()
    provider.model = "ollama/qwen3-coder:480b-cloud"
    full_ctx.agent.llm = provider
    # Should not raise
    await default_registry.dispatch(full_ctx, "/model")


@pytest.mark.asyncio
async def test_model_switch(default_registry, full_ctx):
    """``/model <name>`` should call provider_factory and swap agent.llm."""
    old_provider = MagicMock()
    old_provider.model = "ollama/old-model"
    full_ctx.agent.llm = old_provider

    new_provider = MagicMock()
    full_ctx.provider_factory = MagicMock(return_value=new_provider)

    await default_registry.dispatch(full_ctx, "/model ollama/qwen2.5-coder:14b")
    full_ctx.provider_factory.assert_called_once_with("ollama/qwen2.5-coder:14b")
    assert full_ctx.agent.llm is new_provider
