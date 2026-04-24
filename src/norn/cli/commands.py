"""Slash command system for Norn's interactive chat mode."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable

if TYPE_CHECKING:
    from rich.console import Console

    from norn.core.agent import AgentLoop
    from norn.core.config import NornConfig
    from norn.flags.registry import FeatureFlagRegistry


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


class _ExitRequested(Exception):
    """Raised by /exit to signal the chat loop to stop."""


async def _show_models(ctx: CommandContext) -> None:
    """Display current model, router tiers, and local Ollama models."""
    # Current model
    model_name = getattr(ctx.agent, "llm", None)
    if model_name is not None:
        model_name = getattr(model_name, "model", None)
    ctx.console.print(f"[bold]Current model:[/bold] {model_name or 'unknown'}")

    # Router tiers
    if ctx.config and hasattr(ctx.config, "router") and ctx.config.router:
        tiers = getattr(ctx.config.router, "tiers", None)
        if tiers:
            ctx.console.print("[bold]Router tiers:[/bold]")
            for tier_name, tier_cfg in tiers.items():
                ctx.console.print(f"  {tier_name}: {tier_cfg.provider}/{tier_cfg.model}")

    # Best-effort Ollama local models
    try:
        import httpx

        resp = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        if resp.status_code == 200:
            data = resp.json()
            models = [m.get("name", "?") for m in data.get("models", [])]
            if models:
                ctx.console.print("[bold]Local Ollama models:[/bold]")
                for m in models:
                    ctx.console.print(f"  {m}")
    except Exception:  # noqa: BLE001
        pass


def _register_session_commands(reg: SlashCommandRegistry) -> None:
    """Register /help, /exit, /clear, /model commands."""

    @slash_command("/help", description="Show available commands", registry=reg)
    async def cmd_help(ctx: CommandContext, args: str) -> None:
        ctx.console.print(reg.help_text())

    @slash_command("/exit", description="Exit the chat session", registry=reg)
    async def cmd_exit(ctx: CommandContext, args: str) -> None:
        raise _ExitRequested()

    @slash_command("/clear", description="Clear conversation history", registry=reg)
    async def cmd_clear(ctx: CommandContext, args: str) -> None:
        ctx.agent.history = []
        ctx.agent.user_message_count = 0
        ctx.agent.tool_call_count = 0
        ctx.console.print("[green]History cleared.[/green]")

    @slash_command("/model", description="List models or switch provider", registry=reg)
    async def cmd_model(ctx: CommandContext, args: str) -> None:
        if not args.strip():
            await _show_models(ctx)
            return
        model_str = args.strip()
        if ctx.provider_factory is None:
            ctx.console.print("[red]No provider factory available.[/red]")
            return
        try:
            new_provider = ctx.provider_factory(model_str)
            ctx.agent.llm = new_provider
            ctx.console.print(f"[green]Switched to {model_str}[/green]")
        except Exception as exc:  # noqa: BLE001
            ctx.console.print(f"[red]Failed to switch model: {exc}[/red]")


def _register_debug_commands(reg: SlashCommandRegistry) -> None:
    """Register /tokens, /history, /tools debug commands."""

    @slash_command("/tokens", description="Show session token/message stats", registry=reg)
    async def cmd_tokens(ctx: CommandContext, args: str) -> None:
        agent = ctx.agent
        ctx.console.print(f"[bold]Session stats:[/bold]")
        ctx.console.print(f"  User messages: {agent.user_message_count}")
        ctx.console.print(f"  Tool calls:    {agent.tool_call_count}")
        ctx.console.print(f"  History length: {len(agent.history)}")

    @slash_command("/history", description="Show conversation history", registry=reg)
    async def cmd_history(ctx: CommandContext, args: str) -> None:
        if not ctx.agent.history:
            ctx.console.print("[dim]No messages yet.[/dim]")
            return
        for i, msg in enumerate(ctx.agent.history):
            role = getattr(msg, "role", "?")
            tool_calls = getattr(msg, "tool_calls", None)
            if tool_calls:
                names = ", ".join(getattr(tc, "name", str(tc)) for tc in tool_calls)
                ctx.console.print(f"  [{i}] {role}: [tool_calls: {names}]")
            else:
                content = str(getattr(msg, "content", "") or "")
                preview = content[:80] + ("…" if len(content) > 80 else "")
                ctx.console.print(f"  [{i}] {role}: {preview}")

    @slash_command("/tools", description="List active tools and risk levels", registry=reg)
    async def cmd_tools(ctx: CommandContext, args: str) -> None:
        tools = ctx.agent.registry.list_tools()
        if not tools:
            ctx.console.print("[dim]No tools registered.[/dim]")
            return
        color_map = {"low": "green", "medium": "yellow", "high": "red"}
        for tool in tools:
            risk = tool.risk_level.value
            color = color_map.get(risk, "white")
            ctx.console.print(f"  [{color}]{risk:<6}[/{color}] {tool.name} — {tool.description}")


def _register_config_commands(reg: SlashCommandRegistry) -> None:
    """Register /mode, /flags, /memory config commands."""

    @slash_command("/mode", description="Show or switch permission mode", registry=reg)
    async def cmd_mode(ctx: CommandContext, args: str) -> None:
        from norn.core.config import PermissionMode

        if not args.strip():
            current = ctx.agent.permission_checker.mode
            options = ", ".join(m.value for m in PermissionMode)
            ctx.console.print(f"[bold]Current mode:[/bold] {current.value}")
            ctx.console.print(f"[dim]Available: {options}[/dim]")
            return
        value = args.strip().lower()
        try:
            new_mode = PermissionMode(value)
        except ValueError:
            options = ", ".join(m.value for m in PermissionMode)
            ctx.console.print(f"[red]Invalid mode '{value}'. Choose from: {options}[/red]")
            return
        ctx.agent.permission_checker.mode = new_mode
        ctx.console.print(f"[green]Permission mode set to {new_mode.value}[/green]")

    @slash_command("/flags", description="List or toggle feature flags", registry=reg)
    async def cmd_flags(ctx: CommandContext, args: str) -> None:
        fr = ctx.flag_registry
        if fr is None:
            ctx.console.print("[red]No flag registry available.[/red]")
            return
        if not args.strip():
            flags = fr.list_flags()
            if not flags:
                ctx.console.print("[dim]No feature flags registered.[/dim]")
                return
            for flag in flags:
                enabled = fr.is_enabled(flag.name)
                status = "[green]ON[/green]" if enabled else "[red]OFF[/red]"
                ctx.console.print(f"  {status}  {flag.name} — {flag.description}")
            return
        name = args.strip()
        current = fr.is_enabled(name)
        fr._config_overrides[name] = not current
        new_state = "ON" if not current else "OFF"
        ctx.console.print(f"[green]Flag '{name}' set to {new_state}[/green]")

    @slash_command("/memory", description="Show memory contents", registry=reg)
    async def cmd_memory(ctx: CommandContext, args: str) -> None:
        from rich.markdown import Markdown

        store = ctx.agent.memory_store
        if store is None:
            ctx.console.print("[dim]Memory is not enabled.[/dim]")
            return
        content = store.read_memory()
        if not content:
            ctx.console.print("[dim]Memory is empty.[/dim]")
            return
        ctx.console.print(Markdown(content))


def build_default_registry() -> SlashCommandRegistry:
    """Create a registry with all default commands."""
    reg = SlashCommandRegistry()
    _register_session_commands(reg)
    _register_debug_commands(reg)
    _register_config_commands(reg)
    return reg
