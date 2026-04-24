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
