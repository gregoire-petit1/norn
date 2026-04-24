# Slash Commands System — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add an extensible slash command system to Norn's chat mode with 10 commands, tab autocompletion, and user-friendly error handling.

**Architecture:** A `SlashCommandRegistry` with decorator-based registration. Commands receive a `CommandContext` dataclass giving access to agent, console, config, and a provider factory. The chat loop intercepts `/`-prefixed input before sending to the LLM. `prompt_toolkit`'s `WordCompleter` provides tab completion on command names.

**Tech Stack:** prompt_toolkit (completer), Rich (output formatting), httpx (Ollama model listing), litellm (error types)

---

## Task 1: Core infrastructure — SlashCommandRegistry + CommandContext

**Files:**
- Create: `src/norn/cli/commands.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing tests**

```python
# tests/test_cli/test_commands.py
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
```

**Step 2: Run tests — expect FAIL** (module doesn't exist)

```bash
uv run pytest tests/test_cli/test_commands.py -v --tb=short
```

**Step 3: Implement core**

```python
# src/norn/cli/commands.py
"""Slash command system for Norn's interactive chat mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Awaitable

if TYPE_CHECKING:
    from norn.core.agent import AgentLoop
    from norn.core.config import NornConfig
    from norn.flags.registry import FeatureFlagRegistry
    from rich.console import Console


@dataclass
class CommandContext:
    """Context passed to slash command handlers."""

    agent: AgentLoop | None
    console: Console
    config: NornConfig | None
    provider_factory: Callable | None  # (model_str) -> LLMProvider
    flag_registry: FeatureFlagRegistry | None


# Type alias for command handlers
CommandHandler = Callable[[CommandContext, str], Awaitable[None]]


@dataclass
class _CommandEntry:
    name: str
    description: str
    handler: CommandHandler


class SlashCommandRegistry:
    """Registry for slash commands with dispatch and help generation."""

    def __init__(self) -> None:
        self._commands: dict[str, _CommandEntry] = {}

    def register(self, name: str, description: str, handler: CommandHandler) -> None:
        self._commands[name] = _CommandEntry(name=name, description=description, handler=handler)

    def command_names(self) -> list[str]:
        return sorted(self._commands.keys())

    def help_text(self) -> str:
        lines = []
        for name in sorted(self._commands):
            entry = self._commands[name]
            lines.append(f"  {name:<12} {entry.description}")
        return "\n".join(lines)

    async def dispatch(self, ctx: CommandContext, raw_input: str) -> None:
        parts = raw_input.strip().split(maxsplit=1)
        name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        entry = self._commands.get(name)
        if entry is None:
            ctx.console.print(f"[red]Unknown command: {name}[/red]. Type /help for commands.")
            return

        await entry.handler(ctx, args)


def slash_command(
    name: str,
    description: str,
    registry: SlashCommandRegistry | None = None,
) -> Callable:
    """Decorator to register a slash command."""
    def decorator(fn: CommandHandler) -> CommandHandler:
        if registry is not None:
            registry.register(name, description, fn)
        return fn
    return decorator
```

**Step 4: Run tests — expect PASS**

```bash
uv run pytest tests/test_cli/test_commands.py -v --tb=short
```

**Step 5: Commit**

```bash
git add src/norn/cli/commands.py tests/test_cli/test_commands.py
git commit -m "feat(cli): add SlashCommandRegistry with decorator and CommandContext"
```

---

## Task 2: Session commands — /help, /exit, /clear

**Files:**
- Modify: `src/norn/cli/commands.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing tests**

Append to `tests/test_cli/test_commands.py`:

```python
from norn.cli.commands import build_default_registry
from unittest.mock import AsyncMock, MagicMock


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
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Add to `src/norn/cli/commands.py`:

```python
# Global default registry
_default_registry = SlashCommandRegistry()


def build_default_registry() -> SlashCommandRegistry:
    """Build a registry with all built-in slash commands."""
    reg = SlashCommandRegistry()
    _register_session_commands(reg)
    _register_debug_commands(reg)
    _register_config_commands(reg)
    return reg


class _ExitRequested(Exception):
    """Raised by /exit to signal the chat loop to stop."""


def _register_session_commands(reg: SlashCommandRegistry) -> None:
    @slash_command("/help", description="Show available commands", registry=reg)
    async def cmd_help(ctx: CommandContext, args: str) -> None:
        ctx.console.print("[bold]Available commands:[/bold]\n")
        ctx.console.print(reg.help_text())

    @slash_command("/exit", description="Exit the chat", registry=reg)
    async def cmd_exit(ctx: CommandContext, args: str) -> None:
        raise _ExitRequested()

    @slash_command("/clear", description="Clear conversation history", registry=reg)
    async def cmd_clear(ctx: CommandContext, args: str) -> None:
        if ctx.agent is not None:
            ctx.agent.history.clear()
            ctx.agent.user_message_count = 0
            ctx.agent.tool_call_count = 0
        ctx.console.print("[dim]History cleared.[/dim]")
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add /help, /exit, /clear session commands"
```

---

## Task 3: /model command — list + switch

**Files:**
- Modify: `src/norn/cli/commands.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing tests**

```python
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
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

In `_register_session_commands`:

```python
@slash_command("/model", description="List available models or switch model", registry=reg)
async def cmd_model(ctx: CommandContext, args: str) -> None:
    if not args.strip():
        _show_models(ctx)
        return

    model_str = args.strip()
    if ctx.provider_factory is None or ctx.agent is None:
        ctx.console.print("[red]Cannot switch model in this context.[/red]")
        return

    try:
        new_provider = ctx.provider_factory(model_str)
        ctx.agent.llm = new_provider
        ctx.console.print(f"[green]Switched to:[/green] {model_str}")
    except Exception as e:
        ctx.console.print(f"[red]Failed to switch model: {e}[/red]")


def _show_models(ctx: CommandContext) -> None:
    """Show current model and configured models."""
    # Current model
    current = "unknown"
    if ctx.agent is not None and hasattr(ctx.agent.llm, "model"):
        current = ctx.agent.llm.model
    ctx.console.print(f"[bold]Current model:[/bold] {current}\n")

    # Router tiers from config
    if ctx.config is not None and ctx.config.router.tiers:
        ctx.console.print("[bold]Router tiers:[/bold]")
        for tier_name, tier_cfg in ctx.config.router.tiers.items():
            prefix = "ollama/" if tier_cfg.provider == "ollama" else (
                "openrouter/" if tier_cfg.provider == "openrouter" else ""
            )
            ctx.console.print(f"  {tier_name:<12} {prefix}{tier_cfg.model}")
        ctx.console.print()

    # Query Ollama for local models (best-effort)
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m["name"] for m in data.get("models", [])]
            if models:
                ctx.console.print("[bold]Ollama local models:[/bold]")
                for m in sorted(models):
                    ctx.console.print(f"  ollama/{m}")
    except Exception:
        pass  # Ollama not running or unreachable
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add /model command — list models and switch provider"
```

---

## Task 4: Debug commands — /tokens, /history, /tools

**Files:**
- Modify: `src/norn/cli/commands.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing tests**

```python
@pytest.fixture
def debug_ctx():
    agent = MagicMock()
    agent.history = [
        MagicMock(role="user", content="hello", tool_calls=None),
        MagicMock(role="assistant", content="hi there!", tool_calls=None),
    ]
    agent.user_message_count = 1
    agent.tool_call_count = 3

    registry = MagicMock()
    registry.list_tools.return_value = []

    return CommandContext(
        agent=agent,
        console=Console(file=None, force_terminal=False, no_color=True, width=120),
        config=None,
        provider_factory=None,
        flag_registry=None,
    )


def test_default_registry_has_debug_commands(default_registry):
    names = default_registry.command_names()
    assert "/tokens" in names
    assert "/history" in names
    assert "/tools" in names


@pytest.mark.asyncio
async def test_tokens_shows_counts(default_registry, debug_ctx):
    """``/tokens`` should not raise."""
    await default_registry.dispatch(debug_ctx, "/tokens")


@pytest.mark.asyncio
async def test_history_shows_messages(default_registry, debug_ctx):
    """``/history`` should not raise."""
    await default_registry.dispatch(debug_ctx, "/history")
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

```python
def _register_debug_commands(reg: SlashCommandRegistry) -> None:
    @slash_command("/tokens", description="Show token usage for this session", registry=reg)
    async def cmd_tokens(ctx: CommandContext, args: str) -> None:
        if ctx.agent is None:
            ctx.console.print("[dim]No agent context.[/dim]")
            return
        ctx.console.print(f"[bold]Session stats:[/bold]")
        ctx.console.print(f"  Messages: {ctx.agent.user_message_count}")
        ctx.console.print(f"  Tool calls: {ctx.agent.tool_call_count}")
        ctx.console.print(f"  History length: {len(ctx.agent.history)} messages")

    @slash_command("/history", description="Show conversation history", registry=reg)
    async def cmd_history(ctx: CommandContext, args: str) -> None:
        if ctx.agent is None or not ctx.agent.history:
            ctx.console.print("[dim]No history.[/dim]")
            return
        for i, msg in enumerate(ctx.agent.history):
            role = msg.role
            content = (msg.content or "")[:80]
            if msg.tool_calls:
                tool_names = ", ".join(tc.name for tc in msg.tool_calls)
                content = f"[tools: {tool_names}]"
            ctx.console.print(f"  {i:>3}. [{role}] {content}")

    @slash_command("/tools", description="List active tools", registry=reg)
    async def cmd_tools(ctx: CommandContext, args: str) -> None:
        if ctx.agent is None:
            ctx.console.print("[dim]No agent context.[/dim]")
            return
        tools = ctx.agent.registry.list_tools()
        if not tools:
            ctx.console.print("[dim]No tools registered.[/dim]")
            return
        risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
        for tool in tools:
            color = risk_colors.get(tool.risk_level.value, "white")
            ctx.console.print(
                f"  [{color}]{tool.risk_level.value:>6}[/{color}]  "
                f"[bold]{tool.name}[/bold] - {tool.description}"
            )
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add /tokens, /history, /tools debug commands"
```

---

## Task 5: Config commands — /mode, /flags, /memory

**Files:**
- Modify: `src/norn/cli/commands.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing tests**

```python
from norn.core.config import NornConfig, PermissionMode


@pytest.fixture
def config_ctx():
    agent = MagicMock()
    agent.permission_checker = MagicMock()
    agent.permission_checker.mode = PermissionMode.INTERACTIVE
    agent.memory_store = None
    config = NornConfig()

    flag_registry = MagicMock()
    flag_registry.list_flags.return_value = [
        MagicMock(name="ml_tools", default=True, description="ML tools"),
    ]
    flag_registry.is_enabled.return_value = True

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
    """``/mode`` without args should show current mode."""
    await default_registry.dispatch(config_ctx, "/mode")


@pytest.mark.asyncio
async def test_mode_switch(default_registry, config_ctx):
    """``/mode yolo`` should switch permission mode."""
    await default_registry.dispatch(config_ctx, "/mode yolo")
    assert config_ctx.agent.permission_checker.mode == PermissionMode.YOLO


@pytest.mark.asyncio
async def test_flags_shows_list(default_registry, config_ctx):
    """``/flags`` should not raise."""
    await default_registry.dispatch(config_ctx, "/flags")
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

```python
def _register_config_commands(reg: SlashCommandRegistry) -> None:
    @slash_command("/mode", description="Show/switch permission mode (yolo/auto/interactive/strict)", registry=reg)
    async def cmd_mode(ctx: CommandContext, args: str) -> None:
        if ctx.agent is None or ctx.agent.permission_checker is None:
            ctx.console.print("[dim]No permission checker configured.[/dim]")
            return

        if not args.strip():
            current = ctx.agent.permission_checker.mode.value
            ctx.console.print(f"[bold]Permission mode:[/bold] {current}")
            ctx.console.print(f"  Options: interactive, auto, yolo, strict")
            return

        from norn.core.config import PermissionMode
        mode_str = args.strip().lower()
        try:
            new_mode = PermissionMode(mode_str)
        except ValueError:
            ctx.console.print(f"[red]Unknown mode: {mode_str}[/red]. Options: interactive, auto, yolo, strict")
            return
        ctx.agent.permission_checker.mode = new_mode
        ctx.console.print(f"[green]Permission mode set to:[/green] {new_mode.value}")

    @slash_command("/flags", description="Show/toggle feature flags", registry=reg)
    async def cmd_flags(ctx: CommandContext, args: str) -> None:
        if ctx.flag_registry is None:
            ctx.console.print("[dim]No flag registry.[/dim]")
            return

        if not args.strip():
            ctx.console.print("[bold]Feature flags:[/bold]")
            for flag in ctx.flag_registry.list_flags():
                enabled = ctx.flag_registry.is_enabled(flag.name)
                status = "[green]on[/green]" if enabled else "[red]off[/red]"
                ctx.console.print(f"  {flag.name:<16} {status}  {flag.description}")
            return

        # Toggle: /flags ml_tools
        flag_name = args.strip()
        current = ctx.flag_registry.is_enabled(flag_name)
        ctx.flag_registry._config_overrides[flag_name] = not current
        new_state = "on" if not current else "off"
        ctx.console.print(f"[green]{flag_name}[/green] toggled to {new_state}")

    @slash_command("/memory", description="Show persistent memory contents", registry=reg)
    async def cmd_memory(ctx: CommandContext, args: str) -> None:
        if ctx.agent is None or ctx.agent.memory_store is None:
            ctx.console.print("[dim]Memory system is disabled.[/dim]")
            return
        content = ctx.agent.memory_store.read_memory()
        if content.strip():
            from rich.markdown import Markdown
            ctx.console.print(Markdown(content))
        else:
            ctx.console.print("[dim]Memory is empty.[/dim]")
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add /mode, /flags, /memory config commands"
```

---

## Task 6: Tab autocompletion

**Files:**
- Modify: `src/norn/cli/commands.py`
- Modify: `src/norn/cli/main.py`
- Test: `tests/test_cli/test_commands.py`

**Step 1: Write failing test**

```python
from prompt_toolkit.document import Document
from norn.cli.commands import build_slash_completer


def test_completer_returns_command_names():
    """Completer should suggest command names."""
    reg = build_default_registry()
    completer = build_slash_completer(reg)
    doc = Document(text="/", cursor_position=1)
    completions = list(completer.get_completions(doc, None))
    names = [c.text for c in completions]
    assert "/help" in names
    assert "/model" in names
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

In `src/norn/cli/commands.py`:

```python
def build_slash_completer(registry: SlashCommandRegistry):
    """Build a prompt_toolkit completer for slash commands."""
    from prompt_toolkit.completion import WordCompleter
    return WordCompleter(
        registry.command_names(),
        sentence=True,  # Treat whole command as a word
    )
```

In `src/norn/cli/main.py`, update the PromptSession:

```python
from norn.cli.commands import build_default_registry, build_slash_completer, CommandContext, _ExitRequested

cmd_registry = build_default_registry()
completer = build_slash_completer(cmd_registry)

session = PromptSession(
    message="> ",
    multiline=False,
    key_bindings=bindings,
    completer=completer,
)
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add tab autocompletion for slash commands"
```

---

## Task 7: Error 429 user-friendly handling

**Files:**
- Modify: `src/norn/cli/main.py` (or `src/norn/cli/renderer.py`)
- Test: `tests/test_cli/test_renderer.py`

**Step 1: Write failing test**

```python
# In tests/test_cli/test_renderer.py or tests/test_cli/test_commands.py

from norn.cli.errors import format_llm_error


def test_format_429_error():
    """429 errors should produce a user-friendly message."""
    err = Exception('OllamaException - {"StatusCode":429,"Status":"429 Too Many Requests","error":"rate limit"}')
    msg = format_llm_error(err)
    assert "rate limit" in msg.lower() or "Rate limit" in msg


def test_format_connection_error():
    """Connection errors should produce a user-friendly message."""
    err = ConnectionError("Connection refused")
    msg = format_llm_error(err)
    assert "connect" in msg.lower()


def test_format_generic_error():
    """Unknown errors should pass through."""
    err = Exception("something weird")
    msg = format_llm_error(err)
    assert "something weird" in msg
```

**Step 2: Run tests — expect FAIL**

**Step 3: Implement**

Create `src/norn/cli/errors.py`:

```python
"""User-friendly error formatting for CLI."""

from __future__ import annotations

import re

_RATE_LIMIT_PATTERNS = (
    re.compile(r"429", re.IGNORECASE),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
)

_CONNECTION_PATTERNS = (
    re.compile(r"connection.?(refused|error|reset)", re.IGNORECASE),
    re.compile(r"unreachable", re.IGNORECASE),
)


def format_llm_error(exc: Exception) -> str:
    """Format an LLM error into a user-friendly message."""
    msg = str(exc)

    for pattern in _RATE_LIMIT_PATTERNS:
        if pattern.search(msg):
            return "Rate limit reached. Wait a few minutes or use /model to switch provider."

    for pattern in _CONNECTION_PATTERNS:
        if pattern.search(msg):
            return "Cannot connect to the LLM provider. Check that it's running."

    if "timeout" in msg.lower():
        return "Request timed out. The model may be overloaded."

    # Fallback: clean up the raw error
    return f"LLM error: {msg}"
```

Update `src/norn/cli/main.py` chat loop to use it:

```python
from norn.cli.errors import format_llm_error

# In the chat loop, replace:
#     except Exception as e:
#         console.print(f"[red]Error: {e}[/red]")
# with:
            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                console.print(f"[red]{format_llm_error(e)}[/red]")
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(cli): add user-friendly LLM error formatting (429, connection, timeout)"
```

---

## Task 8: Wire everything into chat loop

**Files:**
- Modify: `src/norn/cli/main.py`
- Test: manual smoke test

**Step 1: Update chat command**

Replace the hardcoded `exit`/`quit` check and wire the command system:

```python
# In chat() command, after building agent:

from norn.cli.commands import (
    build_default_registry,
    build_slash_completer,
    CommandContext,
    _ExitRequested,
)
from norn.cli.errors import format_llm_error

cmd_registry = build_default_registry()
cmd_ctx = CommandContext(
    agent=agent,
    console=console,
    config=config,
    provider_factory=lambda model_str: build_litellm_provider(
        model_str.split("/")[0] if "/" in model_str else "ollama",
        model_str.split("/", 1)[1] if "/" in model_str else model_str,
        api_base=None,
    ),
    flag_registry=flag_registry,
)
completer = build_slash_completer(cmd_registry)

# In the chat loop:

async def _chat_loop() -> None:
    renderer = StreamRenderer(console)
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _insert_newline(event):
        event.current_buffer.insert_text("\n")

    session = PromptSession(
        message="> ",
        multiline=False,
        key_bindings=bindings,
        completer=completer,
    )

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(None, session.prompt)
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye.")
            break

        if not user_input.strip():
            continue

        # Slash command dispatch
        if user_input.strip().startswith("/"):
            try:
                await cmd_registry.dispatch(cmd_ctx, user_input.strip())
            except _ExitRequested:
                console.print("Goodbye.")
                break
            continue

        # LLM turn
        try:
            await renderer.render(agent.run_stream(user_input))
            console.print()
        except KeyboardInterrupt:
            console.print("\n[dim]Interrupted.[/dim]")
        except Exception as e:
            console.print(f"[red]{format_llm_error(e)}[/red]")
```

**Step 2: Remove old `exit`/`quit` hardcoded check** (now handled by `/exit`)

Keep backward compat: also treat bare `exit`/`quit` as `/exit` for convenience.

**Step 3: Run full test suite**

```bash
uv run pytest --tb=short -q
```

**Step 4: Commit**

```bash
git commit -m "feat(cli): wire slash commands, tab completion, error handling into chat loop"
```

---

## Execution Order

| Task | Depends On | Est. Time |
|------|-----------|-----------|
| 1. Core infra | — | 15 min |
| 2. Session commands | 1 | 10 min |
| 3. /model command | 1 | 15 min |
| 4. Debug commands | 1 | 10 min |
| 5. Config commands | 1 | 15 min |
| 6. Tab autocompletion | 1 | 10 min |
| 7. Error formatting | — | 10 min |
| 8. Chat loop wiring | 1-7 | 15 min |
| **Total** | | **~1h40** |
