"""Norn CLI entrypoint."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt

from norn.cli.logs import logs_app
from norn.core.agent import AgentLoop
from norn.core.config import NornConfig
from norn.core.router import RouterProvider, Tier, build_litellm_provider
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier

if TYPE_CHECKING:
    from norn.core.llm import LiteLLMProvider
    from norn.permissions.models import PermissionRequest

from norn.memory.models import MemoryConfig
from norn.memory.session_logger import SessionLogger
from norn.memory.store import MemoryStore
from norn.observability import init_logging, new_session
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.ml.dataset_inspector import DatasetInspectorTool
from norn.tools.ml.model_card import ModelCardTool
from norn.tools.ml.model_eval import ModelEvalTool
from norn.tools.ml.model_inspector import ModelInspectorTool
from norn.tools.ml.tensor_inspector import TensorInspectorTool
from norn.tools.registry import ToolRegistry
from norn.tools.web.web_fetch import WebFetchTool
from norn.tools.web.web_search import WebSearchTool

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
app.add_typer(logs_app, name="logs")
console = Console()

# Reusable CLI option for forcing a router tier.
_TIER_NAMES_HELP = ", ".join(t.value for t in Tier)
ModelOption = Annotated[
    str | None,
    typer.Option(
        "--model",
        help=f"Force routing tier: {_TIER_NAMES_HELP} (router mode only)",
    ),
]


def _build_registry(flag_registry: FeatureFlagRegistry | None = None) -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    # ML tools (gated behind ml_tools feature flag)
    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")
    # Web tools (gated behind web_search feature flag)
    registry.register(WebFetchTool(), feature_flag="web_search")
    registry.register(WebSearchTool(), feature_flag="web_search")
    return registry


def _build_provider(
    config: NornConfig, model_override: str | None = None
) -> LiteLLMProvider | RouterProvider:
    """Build the LLM provider from config.

    If `config.router.enabled`, returns a `RouterProvider` (3-tier routing with fallback);
    otherwise returns a `LiteLLMProvider` using the legacy `config.llm` block.

    `model_override` (e.g. "fast", "standard", "powerful") forces the router's default tier
    when the router is enabled. Ignored otherwise.
    """
    if config.router.enabled:
        default_tier: Tier | None = None
        if model_override is not None:
            try:
                default_tier = Tier(model_override.lower())
            except ValueError:
                console.print(
                    f"[yellow]Unknown tier '{model_override}'. "
                    f"Valid: {_TIER_NAMES_HELP}. Ignoring.[/yellow]"
                )
        return RouterProvider(config.router, default_tier=default_tier)

    # Legacy single-provider path
    return build_litellm_provider(config.llm.provider, config.llm.model, config.llm.api_base)


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
            "web_search": FeatureFlag("web_search", False, "Web search and fetch"),
            "mcp": FeatureFlag("mcp", False, "MCP server tools"),
        }
    )
    registry.apply_config(
        {
            "dream_system": config.flags.dream_system,
            "coordinator": config.flags.coordinator,
            "ml_tools": config.flags.ml_tools,
            "web_search": config.flags.web_search,
            "mcp": config.flags.mcp,
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


def _load_mcp_adapters(config: NornConfig, flag_registry: FeatureFlagRegistry) -> list:
    """Synchronously load MCP tool adapters if the mcp flag is enabled."""
    if not flag_registry.is_enabled("mcp") or not config.mcp.servers:
        return []
    try:
        from norn.mcp.pool import load_mcp_tools

        return asyncio.run(load_mcp_tools(config.mcp))
    except Exception as exc:
        console.print(f"[yellow]MCP load warning: {exc}[/yellow]")
        return []


def _bootstrap_logging(config: NornConfig, verbose: bool = False) -> None:
    """Initialize structured logging and bind a fresh session UUID.

    Must be called at the start of every CLI command (before any work starts)
    so that every subsequent log event carries a session_id.

    Resolution order for log level:
      --verbose flag (forces DEBUG)  >  NORN_LOG_LEVEL env  >  config.logging.level
    """
    init_logging(
        config.logging,
        cli_level_override="DEBUG" if verbose else None,
    )
    new_session()


VerboseOption = Annotated[
    bool,
    typer.Option("--verbose", "-v", help="Enable DEBUG level structured logging"),
]


@app.command()
def chat(model: ModelOption = None, verbose: VerboseOption = False) -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    flag_registry = _build_flag_registry(config)
    provider = _build_provider(config, model_override=model)
    registry = _build_registry(flag_registry)
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        try:
            registry.register(adapter)
        except ValueError:
            pass  # Ignore duplicate tool names across servers
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
def run(
    prompt: str = typer.Argument(help="One-shot prompt to execute"),
    model: ModelOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Run a one-shot prompt and exit."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    flag_registry = _build_flag_registry(config)
    provider = _build_provider(config, model_override=model)
    registry = _build_registry(flag_registry)
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        try:
            registry.register(adapter)
        except ValueError:
            pass  # Ignore duplicate tool names across servers
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
def dream(model: ModelOption = None, verbose: VerboseOption = False) -> None:
    """Manually trigger a memory consolidation dream."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    store = _build_memory_store(config)
    if store is None:
        console.print("[yellow]Memory system is disabled.[/yellow]")
        raise typer.Exit(1)

    provider = _build_provider(config, model_override=model)

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
def coordinate(
    prompt: str = typer.Argument(help="Task to coordinate"),
    model: ModelOption = None,
    verbose: VerboseOption = False,
) -> None:
    """Run a task using multi-agent coordinator mode."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    if not config.coordinator.enabled:
        console.print(
            "[yellow]Coordinator is disabled. "
            "Enable with coordinator.enabled=true in config.[/yellow]"
        )
        raise typer.Exit(1)

    provider = _build_provider(config, model_override=model)
    flag_registry = _build_flag_registry(config)
    registry = _build_registry(flag_registry)
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        try:
            registry.register(adapter)
        except ValueError:
            pass  # Ignore duplicate tool names across servers

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
def tools(verbose: VerboseOption = False) -> None:
    """List available tools."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)
    flag_registry = _build_flag_registry(config)
    registry = _build_registry(flag_registry)
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        try:
            registry.register(adapter)
        except ValueError:
            pass  # Ignore duplicate tool names across servers
    console.print("[bold]Available tools:[/bold]\n")
    for tool in registry.list_tools():
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[tool.risk_level.value]
        console.print(
            f"  [{risk_color}]{tool.risk_level.value:>6}[/{risk_color}]  "
            f"[bold]{tool.name}[/bold] - {tool.description}"
        )


@app.command()
def config(verbose: VerboseOption = False) -> None:
    """Show current configuration."""
    cfg = NornConfig.load()
    cfg.apply_env_overrides()
    _bootstrap_logging(cfg, verbose=verbose)
    console.print("[bold]Current configuration:[/bold]\n")
    console.print(f"  LLM provider: {cfg.llm.provider}")
    console.print(f"  LLM model:    {cfg.llm.model}")
    console.print(f"  Permissions:  {cfg.permissions.mode.value}")
    console.print(f"  Dream:        {'enabled' if cfg.flags.dream_system else 'disabled'}")
    console.print(f"  Coordinator:  {'enabled' if cfg.flags.coordinator else 'disabled'}")
    console.print(f"  ML tools:     {'enabled' if cfg.flags.ml_tools else 'disabled'}")
    console.print(f"  Web search:   {'enabled' if cfg.flags.web_search else 'disabled'}")
    console.print(f"  MCP:          {'enabled' if cfg.flags.mcp else 'disabled'}")
    if cfg.mcp.enabled and cfg.mcp.servers:
        console.print(f"  MCP servers:  {len(cfg.mcp.servers)}")
    console.print(f"  Memory:       {'enabled' if cfg.memory.enabled else 'disabled'}")
    if cfg.memory.enabled:
        console.print(f"  Memory dir:   {cfg.memory.memory_dir}")
    if cfg.coordinator.enabled:
        console.print(f"  Coord threshold: {cfg.coordinator.activation_threshold}")
        console.print(f"  Max workers/phase: {cfg.coordinator.max_workers_per_phase}")


@app.command()
def version() -> None:
    """Show Norn version (no config load, no logging init)."""
    console.print("norn 0.1.0")


if __name__ == "__main__":
    app()
