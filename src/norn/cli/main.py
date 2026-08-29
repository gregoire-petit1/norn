"""Norn CLI entrypoint."""

from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.prompt import Confirm

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from norn.cli.banner import print_banner
from norn.cli.bench import bench_app
from norn.cli.commands import (
    CommandContext,
    _ExitRequested,
    build_default_registry,
    build_slash_completer,
)
from norn.cli.errors import format_llm_error
from norn.cli.logs import logs_app
from norn.cli.renderer import StreamRenderer
from norn.core.agent import AgentLoop
from norn.core.config import NornConfig
from norn.core.context import ContextManager
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
from norn.tools.rag.coderag_context import CodeRAGContextTool
from norn.tools.rag.coderag_explain import CodeRAGExplainTool
from norn.tools.rag.coderag_search import CodeRAGSearchTool
from norn.tools.registry import ToolRegistry
from norn.tools.web.web_fetch import WebFetchTool
from norn.tools.web.web_search import WebSearchTool

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
app.add_typer(bench_app, name="bench")
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


def _resolve_vision_model(config: NornConfig) -> tuple[str, str | None]:
    """Resolve the litellm model id (+api_base) for the image_read vision sub-call.

    Uses the router's ``standard`` tier when routing is on (that's the model the
    bench runs against), else the legacy ``config.llm`` block.
    """
    from norn.core.router import prefixed_model_id

    if config.router.enabled and config.router.tiers:
        tier = config.router.tiers.get("standard") or next(iter(config.router.tiers.values()))
        return prefixed_model_id(tier.provider, tier.model), tier.api_base
    return prefixed_model_id(config.llm.provider, config.llm.model), config.llm.api_base


def _build_task_notes(config: NornConfig):
    """Build a TaskNotes instance when the flag is on (wave 2 A1), else None."""
    if not config.agent.task_notes:
        return None
    from norn.core.task_notes import TaskNotes

    path = Path(config.agent.task_notes_path)
    if not path.is_absolute():
        path = Path.cwd() / path
    return TaskNotes(path, max_chars=config.agent.task_notes_max_chars)


def _build_registry(
    flag_registry: FeatureFlagRegistry | None = None,
    *,
    vision_model: str | None = None,
    vision_api_base: str | None = None,
    config: NornConfig | None = None,
    task_notes=None,
) -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry(flag_registry=flag_registry)
    # SOTA v2 (workstream C): fail-closed bash sandbox, resolved through the
    # flag registry (env NORN_FLAG_SANDBOX > config.sandbox.enabled > off).
    sandbox_enabled = flag_registry.is_enabled("sandbox") if flag_registry else False
    sandbox_cfg = config.sandbox if config is not None else None
    registry.register(
        BashTool(
            sandbox_enabled=sandbox_enabled,
            default_policy=sandbox_cfg.default_policy if sandbox_cfg else "workspace-write",
            allow_network=sandbox_cfg.allow_network if sandbox_cfg else False,
            extra_write_paths=sandbox_cfg.extra_write_paths if sandbox_cfg else None,
            escalation_fn=(
                _cli_prompt_fn
                if sandbox_cfg is None or sandbox_cfg.escalation == "prompt"
                else None
            ),
        )
    )
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
    # CodeRAG tools (gated behind coderag feature flag; norn.rag is imported
    # lazily inside execute(), so registration itself is dependency-free)
    registry.register(CodeRAGSearchTool(), feature_flag="coderag")
    registry.register(CodeRAGContextTool(), feature_flag="coderag")
    registry.register(CodeRAGExplainTool(), feature_flag="coderag")
    # Vision tool (gated behind vision_tools feature flag)
    if vision_model:
        from norn.tools.vision.image_read import ImageReadTool

        registry.register(
            ImageReadTool(model=vision_model, api_base=vision_api_base),
            feature_flag="vision_tools",
        )
    # Task notes tool (wave 2 A1): only registered when the scratchpad is on.
    if task_notes is not None:
        from norn.tools.task_notes_tool import TaskNotesTool

        registry.register(TaskNotesTool(task_notes))
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
    return build_litellm_provider(
        config.llm.provider,
        config.llm.model,
        config.llm.api_base,
        prompt_cache=config.llm.prompt_cache,
        max_retries=config.llm.max_retries,
        retry_backoff=config.llm.retry_backoff,
        request_timeout=config.llm.request_timeout,
    )


def _maybe_wrap_recording(
    provider: LiteLLMProvider | RouterProvider,
    config: NornConfig,
    record_flag: bool,
) -> LiteLLMProvider | RouterProvider:
    """Wrap the provider in a RecordingProvider when recording is requested.

    SOTA v2 (workstream A): the recorded JSONL doubles as a replay-test
    fixture. Transparent wrapper — satisfies the LLMProvider protocol.
    """
    if not (record_flag or config.recording.enabled):
        return provider
    from pathlib import Path

    from norn.core.replay import RecordingProvider
    from norn.observability.logger import session_id_var

    sid = session_id_var.get() or "session"
    rec_dir = Path(config.recording.dir).expanduser()
    rec_path = rec_dir / f"{sid}.jsonl"
    console.print(f"[dim]Recording LLM exchanges to {rec_path}[/dim]")
    return RecordingProvider(provider, rec_path)  # type: ignore[return-value]


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
            "coderag": FeatureFlag("coderag", False, "CodeRAG hybrid retrieval tools"),
            "vision_tools": FeatureFlag("vision_tools", False, "image_read multimodal tool"),
            "sandbox": FeatureFlag("sandbox", False, "Fail-closed bash confinement (seatbelt)"),
        }
    )
    registry.apply_config(
        {
            "dream_system": config.flags.dream_system,
            "coordinator": config.flags.coordinator,
            "ml_tools": config.flags.ml_tools,
            "web_search": config.flags.web_search,
            "mcp": config.flags.mcp,
            "coderag": config.flags.coderag,
            "vision_tools": config.flags.vision_tools,
            "sandbox": config.sandbox.enabled,
        }
    )
    return registry


def _build_context_manager(config: NornConfig) -> ContextManager | None:
    """Build context manager if sliding window is enabled."""
    if not config.context.sliding_window:
        return None
    return ContextManager(
        max_history_tokens=config.context.max_history_tokens,
        recent_turns_keep=config.context.recent_turns_keep,
        summary_max_tokens=config.context.summary_max_tokens,
        enabled=True,
    )


def _build_tool_selector(config: NornConfig):
    """Build tool selector if dynamic tool selection is enabled."""
    if not config.context.dynamic_tools:
        return None
    from norn.core.tool_selector import ToolSelector

    return ToolSelector(
        always_include=config.context.always_include_tools,
        max_tools=config.context.max_tools_per_turn,
        enabled=True,
    )


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

RecordOption = Annotated[
    bool,
    typer.Option(
        "--record",
        help="Record LLM exchanges to a session JSONL (replay-test fixture)",
    ),
]


def _print_metrics(response: LLMResponse) -> None:
    """Print a dim metrics line below the response."""
    parts: list[str] = []
    if response.latency_ms is not None:
        parts.append(f"{response.latency_ms / 1000:.1f}s")
    if response.usage is not None:
        parts.append(
            f"{response.usage.prompt_tokens}\u2192{response.usage.completion_tokens} tokens"
        )
    if response.model:
        parts.append(response.model)
    if parts:
        sep = " \u2502 "
        console.print(f"  \u23f1 {sep.join(parts)}", style="dim")


def _make_tool_progress() -> None:
    """Print a dim tool progress line."""

    def _callback(tool_name: str, args_summary: str, duration_ms: int, success: bool) -> None:
        status = "" if success else " [red]FAIL[/red]"
        suffix = f": {args_summary}" if args_summary else ""
        console.print(
            f"  [tool] {tool_name}{suffix} ({duration_ms / 1000:.1f}s){status}",
            style="dim",
        )

    return _callback


@app.command()
def chat(
    model: ModelOption = None,
    verbose: VerboseOption = False,
    record: RecordOption = False,
) -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    flag_registry = _build_flag_registry(config)
    provider = _maybe_wrap_recording(
        _build_provider(config, model_override=model), config, record
    )
    _vision_model, _vision_api_base = _resolve_vision_model(config)
    task_notes = _build_task_notes(config)
    registry = _build_registry(
        flag_registry,
        vision_model=_vision_model,
        vision_api_base=_vision_api_base,
        config=config,
        task_notes=task_notes,
    )
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        with contextlib.suppress(ValueError):
            registry.register(adapter)  # Ignore duplicate tool names across servers
    checker = _build_permission_checker(config)
    memory_store = _build_memory_store(config)
    session_logger = SessionLogger(memory_store) if memory_store else None
    context_manager = _build_context_manager(config)
    tool_selector = _build_tool_selector(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
        memory_store=memory_store,
        session_logger=session_logger,
        max_tool_rounds=config.agent.max_tool_rounds,
        minify_tool_schemas=config.agent.minify_tool_schemas,
        stable_prompt=config.agent.stable_prompt,
        thread_invariants=config.agent.thread_invariants,
        max_tool_result_chars=config.agent.max_tool_result_chars,
        max_turn_output_chars=config.agent.max_turn_output_chars,
        env_bootstrap=config.agent.env_bootstrap,
        repo_map=config.agent.repo_map,
        repo_map_max_chars=config.agent.repo_map_max_chars,
        repo_map_languages=config.agent.repo_map_languages,
        repo_map_exclude=config.agent.repo_map_exclude,
        context_manager=context_manager,
        tool_selector=tool_selector,
        task_notes=task_notes,
    )

    cmd_registry = build_default_registry()
    cmd_ctx = CommandContext(
        agent=agent,
        console=console,
        config=config,
        provider_factory=lambda model_str: build_litellm_provider(
            model_str.split("/")[0] if "/" in model_str else "ollama",
            model_str.split("/", 1)[1] if "/" in model_str else model_str,
            api_base=None,
            max_retries=config.llm.max_retries,
            retry_backoff=config.llm.retry_backoff,
            request_timeout=config.llm.request_timeout,
        ),
        flag_registry=flag_registry,
    )
    completer = build_slash_completer(cmd_registry)

    print_banner(console)
    console.print()
    console.print(f"Permission mode: [bold]{config.permissions.mode.value}[/bold]")
    console.print("Type /help for commands. Alt+Enter for newlines. Ctrl+D to exit.\n")

    async def _chat_loop() -> None:
        renderer = StreamRenderer(console)
        # Build prompt_toolkit session with multiline support
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _insert_newline(event):
            event.current_buffer.insert_text("\n")

        session: PromptSession[str] = PromptSession(
            message="> ",
            multiline=False,  # Enter sends by default
            key_bindings=bindings,
            completer=completer,
        )

        while True:
            try:
                user_input = await asyncio.get_event_loop().run_in_executor(None, session.prompt)
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                break

            if user_input.strip().startswith("/"):
                try:
                    await cmd_registry.dispatch(cmd_ctx, user_input.strip())
                except _ExitRequested:
                    console.print("Goodbye.")
                    break
                continue

            # Backward compat: bare exit/quit
            if user_input.strip().lower() in ("exit", "quit"):
                console.print("Goodbye.")
                break

            if not user_input.strip():
                continue

            try:
                await renderer.render(agent.run_stream(user_input))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                console.print(f"[red]{format_llm_error(e)}[/red]")

    asyncio.run(_chat_loop())


@app.command()
def run(
    prompt: str = typer.Argument(help="One-shot prompt to execute"),
    model: ModelOption = None,
    verbose: VerboseOption = False,
    record: RecordOption = False,
) -> None:
    """Run a one-shot prompt and exit."""
    config = NornConfig.load()
    config.apply_env_overrides()
    _bootstrap_logging(config, verbose=verbose)

    flag_registry = _build_flag_registry(config)
    provider = _maybe_wrap_recording(
        _build_provider(config, model_override=model), config, record
    )
    _vision_model, _vision_api_base = _resolve_vision_model(config)
    task_notes = _build_task_notes(config)
    registry = _build_registry(
        flag_registry,
        vision_model=_vision_model,
        vision_api_base=_vision_api_base,
        config=config,
        task_notes=task_notes,
    )
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        with contextlib.suppress(ValueError):
            registry.register(adapter)  # Ignore duplicate tool names across servers
    checker = _build_permission_checker(config)
    memory_store = _build_memory_store(config)
    session_logger = SessionLogger(memory_store) if memory_store else None
    context_manager = _build_context_manager(config)
    tool_selector = _build_tool_selector(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
        memory_store=memory_store,
        session_logger=session_logger,
        max_tool_rounds=config.agent.max_tool_rounds,
        minify_tool_schemas=config.agent.minify_tool_schemas,
        stable_prompt=config.agent.stable_prompt,
        thread_invariants=config.agent.thread_invariants,
        max_tool_result_chars=config.agent.max_tool_result_chars,
        max_turn_output_chars=config.agent.max_turn_output_chars,
        env_bootstrap=config.agent.env_bootstrap,
        repo_map=config.agent.repo_map,
        repo_map_max_chars=config.agent.repo_map_max_chars,
        repo_map_languages=config.agent.repo_map_languages,
        repo_map_exclude=config.agent.repo_map_exclude,
        context_manager=context_manager,
        tool_selector=tool_selector,
        task_notes=task_notes,
    )

    async def _run_once() -> None:
        try:
            renderer = StreamRenderer(console)
            if config.agent.auto_verify or config.agent.auto_plan:
                await renderer.render(
                    agent.run_verified(
                        prompt,
                        max_verify_rounds=(
                            config.agent.auto_verify_max_rounds
                            if config.agent.auto_verify
                            else 0
                        ),
                        plan_first=config.agent.auto_plan,
                        plan_min_chars=config.agent.auto_plan_min_chars,
                    )
                )
            else:
                await renderer.render(agent.run_stream(prompt))
        except Exception as e:
            console.print(f"[red]{format_llm_error(e)}[/red]")

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
    _vision_model, _vision_api_base = _resolve_vision_model(config)
    registry = _build_registry(
        flag_registry,
        vision_model=_vision_model,
        vision_api_base=_vision_api_base,
        config=config,
    )
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        with contextlib.suppress(ValueError):
            registry.register(adapter)  # Ignore duplicate tool names across servers

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
    _vision_model, _vision_api_base = _resolve_vision_model(config)
    registry = _build_registry(
        flag_registry,
        vision_model=_vision_model,
        vision_api_base=_vision_api_base,
        config=config,
    )
    # Load MCP adapters and inject into registry
    for adapter in _load_mcp_adapters(config, flag_registry):
        with contextlib.suppress(ValueError):
            registry.register(adapter)  # Ignore duplicate tool names across servers
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
