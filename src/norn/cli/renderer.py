"""Streaming CLI renderer for AgentEvents using Rich."""

from __future__ import annotations

from collections.abc import AsyncIterator

from rich.console import Console
from rich.markdown import Markdown

from norn.core.models import AgentEvent, EventType, TokenUsage

# Tool risk-based colors
_TOOL_COLORS: dict[str, str] = {
    "file_read": "green",
    "glob": "green",
    "grep": "green",
    "file_write": "yellow",
    "file_edit": "yellow",
    "bash": "red",
}
_DEFAULT_TOOL_COLOR = "blue"


class StreamRenderer:
    """Renders AgentEvents to the terminal using Rich.

    Accumulates text deltas for final markdown rendering, shows tool
    progress with colored spinners (or static lines in non-TTY), and
    prints usage metrics on DONE.
    """

    def __init__(self, console: Console) -> None:
        self.console = console
        self._text_buffer = ""
        self._spinner = None
        # Observable state for testing
        self.full_text = ""
        self.tool_count = 0
        self.tool_failures = 0
        self.final_usage: TokenUsage | None = None

    async def render(self, events: AsyncIterator[AgentEvent]) -> None:
        """Consume an event stream and render to console."""
        async for event in events:
            match event.type:
                case EventType.TEXT_DELTA:
                    self._handle_text_delta(event)
                case EventType.TOOL_START:
                    self._handle_tool_start(event)
                case EventType.TOOL_END:
                    self._handle_tool_end(event)
                case EventType.DONE:
                    self._handle_done(event)

    def _handle_text_delta(self, event: AgentEvent) -> None:
        """Print text incrementally as raw characters."""
        if event.content:
            self._text_buffer += event.content
            self.full_text += event.content
            self.console.print(event.content, end="", highlight=False)

    def _handle_tool_start(self, event: AgentEvent) -> None:
        """Flush text buffer and show tool spinner or static line."""
        self._flush_markdown()
        color = _TOOL_COLORS.get(event.tool_name or "", _DEFAULT_TOOL_COLOR)
        suffix = f": {event.tool_args}" if event.tool_args else ""
        label = f"  [{color}]{event.tool_name}[/{color}]{suffix}"

        if self.console.is_terminal:
            self._spinner = self.console.status(label, spinner="dots")
            self._spinner.start()
        else:
            # Non-TTY fallback: static line
            self.console.print(f"{label}...", style="dim", end="")

    def _handle_tool_end(self, event: AgentEvent) -> None:
        """Stop spinner and show tool completion status."""
        self.tool_count += 1
        if not event.success:
            self.tool_failures += 1

        # Stop spinner if running
        if self._spinner is not None:
            self._spinner.stop()
            self._spinner = None

        color = _TOOL_COLORS.get(event.tool_name or "", _DEFAULT_TOOL_COLOR)
        status_icon = "ok" if event.success else "[red]FAIL[/red]"
        suffix = f": {event.tool_args}" if event.tool_args else ""
        duration = f" ({event.duration_ms / 1000:.1f}s)" if event.duration_ms else ""

        if self.console.is_terminal:
            # Full line after spinner clears itself
            self.console.print(
                f"  {status_icon} [{color}]{event.tool_name}[/{color}]{suffix}{duration}",
                style="dim",
            )
        else:
            # Inline continuation of the static TOOL_START line
            status = " ok" if event.success else " [red]FAIL[/red]"
            self.console.print(f"{duration}{status}", style="dim")

    def _handle_done(self, event: AgentEvent) -> None:
        """Flush remaining text as markdown and print metrics."""
        self.final_usage = event.usage
        self._flush_markdown()
        self._print_metrics(event)

    def _flush_markdown(self) -> None:
        """Re-render accumulated text buffer as markdown."""
        if not self._text_buffer.strip():
            self._text_buffer = ""
            return
        # Newline to separate from raw streamed text
        self.console.print()
        self.console.print(Markdown(self._text_buffer))
        self._text_buffer = ""

    def _print_metrics(self, event: AgentEvent) -> None:
        """Print a dim metrics line."""
        parts: list[str] = []
        if event.latency_ms:
            parts.append(f"{event.latency_ms / 1000:.1f}s")
        if event.usage:
            parts.append(f"{event.usage.prompt_tokens}\u2192{event.usage.completion_tokens} tokens")
        if event.model:
            parts.append(event.model)
        if parts:
            sep = " \u2502 "
            self.console.print(f"  \u23f1 {sep.join(parts)}", style="dim")
