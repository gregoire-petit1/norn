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
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier

if TYPE_CHECKING:
    from norn.permissions.models import PermissionRequest

from norn.memory.models import MemoryConfig
from norn.memory.session_logger import SessionLogger
from norn.memory.store import MemoryStore
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
console = Console()


def _build_registry(flag_registry: FeatureFlagRegistry | None = None) -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry(flag_registry=flag_registry)
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


def _build_flag_registry(config: NornConfig) -> FeatureFlagRegistry:
    """Build the feature flag registry from config."""
    registry = FeatureFlagRegistry(
        flags={
            "dream_system": FeatureFlag("dream_system", False, "Memory consolidation"),
            "coordinator": FeatureFlag("coordinator", False, "Multi-agent mode"),
            "ml_tools": FeatureFlag("ml_tools", True, "MLOps-specific tools"),
        }
    )
    registry.apply_config(
        {
            "dream_system": config.flags.dream_system,
            "coordinator": config.flags.coordinator,
            "ml_tools": config.flags.ml_tools,
        }
    )
    return registry


def _build_memory_store(config: NornConfig) -> MemoryStore | None:
    """Build memory store if memory is enabled."""
    if not config.memory.enabled:
        return None
    mem_config = MemoryConfig(
        memory_dir=Path(config.memory.memory_dir).expanduser(),
        dream_interval_hours=config.memory.dream_interval_hours,
        dream_min_sessions=config.memory.dream_min_sessions,
    )
    store = MemoryStore(mem_config)
    store.ensure_dirs()
    return store


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()

    flag_registry = _build_flag_registry(config)
    provider = _build_provider(config)
    registry = _build_registry(flag_registry)
    checker = _build_permission_checker(config)
    memory_store = _build_memory_store(config)
    session_logger = SessionLogger(memory_store) if memory_store else None
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
        memory_store=memory_store,
        session_logger=session_logger,
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

    flag_registry = _build_flag_registry(config)
    provider = _build_provider(config)
    registry = _build_registry(flag_registry)
    checker = _build_permission_checker(config)
    memory_store = _build_memory_store(config)
    session_logger = SessionLogger(memory_store) if memory_store else None
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
        memory_store=memory_store,
        session_logger=session_logger,
    )

    async def _run_once() -> None:
        response = await agent.run(prompt)
        if response.content:
            console.print(Markdown(response.content))

    asyncio.run(_run_once())


@app.command()
def dream() -> None:
    """Manually trigger a memory consolidation dream."""
    config = NornConfig.load()
    config.apply_env_overrides()

    store = _build_memory_store(config)
    if store is None:
        console.print("[yellow]Memory system is disabled.[/yellow]")
        raise typer.Exit(1)

    provider = _build_provider(config)

    from norn.dream.engine import DreamEngine

    engine = DreamEngine(store=store, llm=provider)

    async def _dream() -> None:
        with console.status("[dim]Dreaming...[/dim]"):
            result = await engine.dream()
        if result.success:
            console.print(f"[green]Dream complete:[/green] {result.summary}")
            if result.files_written:
                console.print(f"  Written: {', '.join(result.files_written)}")
            if result.files_pruned:
                console.print(f"  Pruned: {', '.join(result.files_pruned)}")
        else:
            console.print(f"[red]Dream failed:[/red] {result.error}")

    asyncio.run(_dream())


@app.command()
def coordinate(prompt: str = typer.Argument(help="Task to coordinate")) -> None:
    """Run a task using multi-agent coordinator mode."""
    config = NornConfig.load()
    config.apply_env_overrides()

    if not config.coordinator.enabled:
        console.print(
            "[yellow]Coordinator is disabled. "
            "Enable with coordinator.enabled=true in config.[/yellow]"
        )
        raise typer.Exit(1)

    provider = _build_provider(config)
    flag_registry = _build_flag_registry(config)
    registry = _build_registry(flag_registry)

    import tempfile

    from norn.coordinator.engine import CoordinatorEngine
    from norn.coordinator.scratchpad import Scratchpad

    scratchpad_dir = Path(tempfile.mkdtemp(prefix="norn-coord-"))
    scratchpad = Scratchpad(base_dir=scratchpad_dir)
    scratchpad.ensure_dirs()

    engine = CoordinatorEngine(
        coordinator_llm=provider,
        worker_llm=provider,
        registry=registry,
        scratchpad=scratchpad,
        cwd=str(Path.cwd()),
    )

    async def _coordinate() -> None:
        with console.status("[dim]Coordinating...[/dim]"):
            result = await engine.coordinate(prompt)

        if result.success:
            console.print(f"[green]Coordinator complete:[/green] {result.summary}")
            for phase, results in result.phase_results.items():
                console.print(f"\n  [bold]{phase.value.upper()}[/bold]:")
                for wr in results:
                    status = "[green]OK[/green]" if wr.success else "[red]FAIL[/red]"
                    console.print(
                        f"    {status} {wr.worker_id}: {(wr.output or wr.error or '')[:80]}"
                    )
        else:
            console.print(f"[red]Coordinator failed:[/red] {result.error}")

    asyncio.run(_coordinate())


@app.command()
def tools() -> None:
    """List available tools."""
    config = NornConfig.load()
    config.apply_env_overrides()
    flag_registry = _build_flag_registry(config)
    registry = _build_registry(flag_registry)
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
    console.print(f"  Memory:       {'enabled' if cfg.memory.enabled else 'disabled'}")
    if cfg.memory.enabled:
        console.print(f"  Memory dir:   {cfg.memory.memory_dir}")
    if cfg.coordinator.enabled:
        console.print(f"  Coord threshold: {cfg.coordinator.activation_threshold}")
        console.print(f"  Max workers/phase: {cfg.coordinator.max_workers_per_phase}")


@app.command()
def version() -> None:
    """Show Norn version."""
    console.print("norn 0.1.0")


if __name__ == "__main__":
    app()
