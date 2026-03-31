"""Norn CLI entrypoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt

from norn.core.agent import AgentLoop
from norn.core.config import NornConfig
from norn.core.llm import LiteLLMProvider
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier

if TYPE_CHECKING:
    from norn.permissions.models import PermissionRequest

from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
console = Console()


def _build_registry() -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


def _build_provider(config: NornConfig) -> LiteLLMProvider:
    """Build the LLM provider from config."""
    model = config.llm.model
    if config.llm.provider == "ollama":
        model = f"ollama/{config.llm.model}"
    elif config.llm.provider == "openrouter":
        model = f"openrouter/{config.llm.model}"
    return LiteLLMProvider(model=model, api_base=config.llm.api_base)


async def _cli_prompt_fn(request: PermissionRequest, description: str) -> bool:
    """Prompt the user for permission approval via rich."""
    risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
    color = risk_colors.get(request.risk_level, "white")
    console.print(
        f"\n[bold {color}]Permission required[/bold {color}]: "
        f"[{color}]{request.tool_name}[/{color}] (risk: {request.risk_level})"
    )
    console.print(f"  {description}")
    return Confirm.ask("  Allow?", default=True)


def _build_permission_checker(config: NornConfig) -> PermissionChecker:
    """Build the permission checker from config."""
    return PermissionChecker(
        mode=config.permissions.mode,
        classifier=RiskClassifier(),
        prompt_fn=_cli_prompt_fn,
    )


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    checker = _build_permission_checker(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
    )

    console.print("[bold]Norn[/bold] - the coding agent that weaves your destiny")
    console.print(f"Permission mode: [bold]{config.permissions.mode.value}[/bold]")
    console.print("Type 'exit' or 'quit' to leave. Ctrl+C to interrupt.\n")

    async def _chat_loop() -> None:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                break

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("Goodbye.")
                break

            if not user_input.strip():
                continue

            try:
                with console.status("[dim]Thinking...[/dim]"):
                    response = await agent.run(user_input)

                if response.content:
                    console.print(Markdown(response.content))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_chat_loop())


@app.command()
def run(prompt: str = typer.Argument(help="One-shot prompt to execute")) -> None:
    """Run a one-shot prompt and exit."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    checker = _build_permission_checker(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
    )

    async def _run_once() -> None:
        response = await agent.run(prompt)
        if response.content:
            console.print(Markdown(response.content))

    asyncio.run(_run_once())


@app.command()
def tools() -> None:
    """List available tools."""
    registry = _build_registry()
    console.print("[bold]Available tools:[/bold]\n")
    for tool in registry.list_tools():
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[tool.risk_level.value]
        console.print(
            f"  [{risk_color}]{tool.risk_level.value:>6}[/{risk_color}]  "
            f"[bold]{tool.name}[/bold] - {tool.description}"
        )


@app.command()
def config() -> None:
    """Show current configuration."""
    cfg = NornConfig.load()
    cfg.apply_env_overrides()
    console.print("[bold]Current configuration:[/bold]\n")
    console.print(f"  LLM provider: {cfg.llm.provider}")
    console.print(f"  LLM model:    {cfg.llm.model}")
    console.print(f"  Permissions:  {cfg.permissions.mode.value}")
    console.print(f"  Dream:        {'enabled' if cfg.flags.dream_system else 'disabled'}")
    console.print(f"  Coordinator:  {'enabled' if cfg.flags.coordinator else 'disabled'}")


@app.command()
def version() -> None:
    """Show Norn version."""
    console.print("norn 0.1.0")


if __name__ == "__main__":
    app()
