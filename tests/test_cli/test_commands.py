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


def test_default_registry_has_debug_commands(default_registry):
    names = default_registry.command_names()
    assert "/tokens" in names
    assert "/history" in names
    assert "/tools" in names


@pytest.mark.asyncio
async def test_tokens_shows_counts(default_registry, full_ctx):
    """``/tokens`` should not raise and show session stats."""
    full_ctx.agent.user_message_count = 5
    full_ctx.agent.tool_call_count = 12
    full_ctx.agent.history = [MagicMock()] * 10
    await default_registry.dispatch(full_ctx, "/tokens")


@pytest.mark.asyncio
async def test_history_shows_messages(default_registry, full_ctx):
    """``/history`` should not raise."""
    await default_registry.dispatch(full_ctx, "/history")


# --- Config commands tests ---

from norn.core.config import NornConfig, PermissionMode


@pytest.fixture
def config_ctx():
    agent = MagicMock()
    agent.history = []
    agent.user_message_count = 0
    agent.tool_call_count = 0
    agent.permission_checker = MagicMock()
    agent.permission_checker.mode = PermissionMode.INTERACTIVE
    agent.memory_store = None

    config = NornConfig()

    flag_registry = MagicMock()
    flag_registry.list_flags.return_value = [
        MagicMock(name="ml_tools", default=True, description="ML tools"),
    ]
    flag_registry.is_enabled.return_value = True
    flag_registry._config_overrides = {}

    return CommandContext(
        agent=agent,
        console=Console(file=None, force_terminal=False, no_color=True, width=120),
        config=config,
        provider_factory=None,
        flag_registry=flag_registry,
    )


def test_default_registry_has_config_commands(default_registry):
    names = default_registry.command_names()
    assert "/mode" in names
    assert "/flags" in names
    assert "/memory" in names


@pytest.mark.asyncio
async def test_mode_shows_current(default_registry, config_ctx):
    """``/mode`` without args should not raise."""
    await default_registry.dispatch(config_ctx, "/mode")


@pytest.mark.asyncio
async def test_mode_switch(default_registry, config_ctx):
    """``/mode yolo`` should switch permission mode."""
    await default_registry.dispatch(config_ctx, "/mode yolo")
    assert config_ctx.agent.permission_checker.mode == PermissionMode.YOLO


@pytest.mark.asyncio
async def test_mode_invalid(default_registry, config_ctx):
    """``/mode badvalue`` should not crash."""
    await default_registry.dispatch(config_ctx, "/mode badvalue")
    # Should remain unchanged
    assert config_ctx.agent.permission_checker.mode == PermissionMode.INTERACTIVE


@pytest.mark.asyncio
async def test_flags_shows_list(default_registry, config_ctx):
    """``/flags`` should not raise."""
    await default_registry.dispatch(config_ctx, "/flags")


# --- Tab autocompletion tests ---

from prompt_toolkit.document import Document
from norn.cli.commands import build_slash_completer


def test_completer_returns_command_names():
    """Completer should suggest command names when user types /."""
    reg = build_default_registry()
    completer = build_slash_completer(reg)
    doc = Document(text="/", cursor_position=1)
    # WordCompleter needs a CompleteEvent; pass None since WordCompleter doesn't use it
    from unittest.mock import MagicMock

    event = MagicMock()
    completions = list(completer.get_completions(doc, event))
    names = [c.text for c in completions]
    assert "/help" in names
    assert "/model" in names


# --- RAMP command tests ---

from unittest.mock import AsyncMock, patch


async def _noop_render(events):
    """Drain async iterator without rendering."""
    async for _ in events:
        pass


@pytest.fixture
def ramp_ctx():
    """Context with a mock agent for RAMP commands."""
    agent = MagicMock()
    agent.history = [MagicMock()]  # non-empty for /proof

    async def _fake_stream(prompt):
        return
        yield  # make it an async generator

    agent.run_stream = _fake_stream
    return CommandContext(
        agent=agent,
        console=Console(file=None, force_terminal=False, no_color=True, width=120),
        config=None,
        provider_factory=None,
        flag_registry=None,
    )


def test_default_registry_has_ramp_commands(default_registry):
    names = default_registry.command_names()
    assert "/plan" in names
    assert "/proof" in names


@pytest.mark.asyncio
async def test_plan_no_args_shows_usage(default_registry, ramp_ctx):
    """/plan without args should print usage, not call run_stream."""
    called = []
    ramp_ctx.agent.run_stream = lambda p: called.append(p) or (_ for _ in ())
    await default_registry.dispatch(ramp_ctx, "/plan")
    assert called == []


@pytest.mark.asyncio
async def test_plan_with_task_calls_run_stream(default_registry, ramp_ctx):
    """/plan <task> should call run_stream with plan prefix + task."""
    from norn.cli.commands import _PLAN_PREFIX

    async def _noop_gen(prompt):
        return
        yield  # makes it an async generator

    mock_stream = MagicMock(side_effect=_noop_gen)
    ramp_ctx.agent.run_stream = mock_stream

    with patch("norn.cli.renderer.StreamRenderer.render", new=AsyncMock()):
        await default_registry.dispatch(ramp_ctx, "/plan fix the sorting bug")

    mock_stream.assert_called_once()
    call_prompt = mock_stream.call_args[0][0]
    assert call_prompt.startswith(_PLAN_PREFIX)
    assert "fix the sorting bug" in call_prompt


@pytest.mark.asyncio
async def test_proof_no_history_shows_message(default_registry, ramp_ctx):
    """/proof with empty history should not call run_stream."""
    ramp_ctx.agent.history = []
    called = []
    ramp_ctx.agent.run_stream = lambda p: called.append(p) or (_ for _ in ())
    await default_registry.dispatch(ramp_ctx, "/proof")
    assert called == []


@pytest.mark.asyncio
async def test_proof_calls_run_stream_with_verification_prompt(default_registry, ramp_ctx):
    """/proof should call run_stream with the multi-perspective verification prompt."""
    from norn.cli.commands import _PROOF_PROMPT

    async def _noop_gen(prompt):
        return
        yield

    mock_stream = MagicMock(side_effect=_noop_gen)
    ramp_ctx.agent.run_stream = mock_stream

    with patch("norn.cli.renderer.StreamRenderer.render", new=AsyncMock()):
        await default_registry.dispatch(ramp_ctx, "/proof")

    mock_stream.assert_called_once()
    call_prompt = mock_stream.call_args[0][0]
    assert "Test-engineer" in call_prompt
    assert "QA" in call_prompt
    assert "End-user" in call_prompt


# --- /reflect command tests ---


def test_default_registry_has_reflect(default_registry):
    assert "/reflect" in default_registry.command_names()


@pytest.mark.asyncio
async def test_reflect_no_history_shows_message(default_registry, ramp_ctx):
    """/reflect with empty history should not call run_stream."""
    ramp_ctx.agent.history = []
    called = []
    ramp_ctx.agent.run_stream = lambda p: called.append(p) or (_ for _ in ())
    await default_registry.dispatch(ramp_ctx, "/reflect")
    assert called == []


@pytest.mark.asyncio
async def test_reflect_calls_run_stream_with_reflect_prompt(default_registry, ramp_ctx):
    """/reflect should call run_stream with the reflection prompt."""
    from norn.cli.commands import _REFLECT_PROMPT

    async def _noop_gen(prompt):
        return
        yield

    mock_stream = MagicMock(side_effect=_noop_gen)
    ramp_ctx.agent.run_stream = mock_stream

    with patch("norn.cli.renderer.StreamRenderer.render", new=AsyncMock()):
        await default_registry.dispatch(ramp_ctx, "/reflect")

    mock_stream.assert_called_once()
    call_prompt = mock_stream.call_args[0][0]
    assert "Reflect" in call_prompt or "reflect" in call_prompt.lower()


@pytest.mark.asyncio
async def test_reflect_saves_lesson_when_memory_store_available(default_registry, ramp_ctx, tmp_path):
    """/reflect should append lesson to memory_store using history content."""
    from norn.memory.models import MemoryConfig
    from norn.memory.store import MemoryStore
    from norn.core.models import AgentEvent, EventType, Message, Role

    store = MemoryStore(MemoryConfig(memory_dir=str(tmp_path / "mem")))
    store.ensure_dirs()
    ramp_ctx.agent.memory_store = store

    # Pre-populate history: one existing message + the reflection assistant msg
    # (simulating what run_stream appends during the reflect turn)
    ramp_ctx.agent.history = [MagicMock()]  # existing history

    async def _noop_gen(prompt):
        return
        yield

    mock_stream = MagicMock(side_effect=_noop_gen)
    ramp_ctx.agent.run_stream = mock_stream

    async def _render_with_side_effect(self_renderer, events):
        # Simulate run_stream appending the reflection to history
        ramp_ctx.agent.history.append(
            Message(role=Role.ASSISTANT, content="Lesson: always check inputs.")
        )

    with patch("norn.cli.renderer.StreamRenderer.render", new=_render_with_side_effect):
        await default_registry.dispatch(ramp_ctx, "/reflect")

    lessons = store.read_lessons()
    assert "Lesson: always check inputs." in lessons
