"""CLI sub-app: norn bench — run, report, list, diff, validate."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

bench_app = typer.Typer(name="bench", help="Benchmark runner for Norn evaluation tasks")
console = Console()

TasksDirOption = Annotated[
    str,
    typer.Option("--tasks-dir", help="Path to tasks directory"),
]
ResultsDirOption = Annotated[
    str,
    typer.Option("--results-dir", help="Path to results directory"),
]


def _ensure_benchmarks_importable() -> None:
    """Add project root to sys.path so benchmarks package is importable."""
    project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def _default_tasks_dir() -> Path:
    return Path("benchmarks/tasks")


def _default_results_dir() -> Path:
    return Path("benchmarks/results")


@bench_app.command("list")
def bench_list(
    tasks_dir: TasksDirOption = "",
    category: str = typer.Option("", help="Filter by category"),
) -> None:
    """List available benchmark tasks."""
    _ensure_benchmarks_importable()
    from benchmarks.runner.loader import load_tasks

    td = Path(tasks_dir) if tasks_dir else _default_tasks_dir()
    cat_filter = category if category else None
    tasks = load_tasks(tasks_dir=td, category=cat_filter)

    if not tasks:
        console.print("[dim]No tasks found.[/dim]")
        return

    table = Table(title=f"Benchmark Tasks ({len(tasks)})", show_lines=False)
    table.add_column("ID", style="bold")
    table.add_column("Category")
    table.add_column("Description")
    table.add_column("Timeout (s)", justify="right")

    for t in tasks:
        table.add_row(t.id, t.category, t.description, str(t.timeout_seconds))

    console.print(table)


@bench_app.command("validate")
def bench_validate(
    tasks_dir: TasksDirOption = "",
) -> None:
    """Validate all task.yaml files."""
    _ensure_benchmarks_importable()
    import yaml as _yaml
    from benchmarks.runner.models import TaskDef

    td = Path(tasks_dir) if tasks_dir else _default_tasks_dir()

    if not td.is_dir():
        console.print(f"[red]Tasks directory not found: {td}[/red]")
        raise typer.Exit(1)

    valid = 0
    invalid = 0

    for task_yaml in sorted(td.rglob("task.yaml")):
        try:
            with open(task_yaml) as f:
                raw = _yaml.safe_load(f) or {}
            if raw.get("deprecated", False):
                continue
            TaskDef.from_yaml(task_yaml)
            valid += 1
        except Exception as e:
            invalid += 1
            rel = task_yaml.relative_to(td)
            console.print(f"  [red]INVALID[/red] {rel}: {e}")

    if invalid == 0:
        console.print(f"[green]{valid} tasks valid, 0 errors.[/green]")
    else:
        console.print(f"\n[yellow]{valid} valid, {invalid} invalid.[/yellow]")


@bench_app.command("report")
def bench_report(
    results_dir: ResultsDirOption = "",
    output: str = typer.Option("", help="Write markdown report to file"),
) -> None:
    """Show the latest benchmark report."""
    _ensure_benchmarks_importable()
    from benchmarks.runner.persistence import list_reports, load_report
    from benchmarks.runner.reporter import render_markdown_report, render_summary_table

    rd = Path(results_dir) if results_dir else _default_results_dir()
    reports = list_reports(results_dir=rd)

    if not reports:
        console.print("[dim]No results found.[/dim]")
        return

    latest = reports[-1]
    report = load_report(latest)
    console.print(render_summary_table(report))

    if output:
        md = render_markdown_report(report)
        Path(output).write_text(md)
        console.print(f"[green]Markdown report written to {output}[/green]")


@bench_app.command("diff")
def bench_diff(
    run_a: str = typer.Argument(help="Path to first result file"),
    run_b: str = typer.Argument(help="Path to second result file"),
) -> None:
    """Compare two benchmark result files."""
    _ensure_benchmarks_importable()
    from benchmarks.runner.persistence import load_report
    from benchmarks.runner.reporter import detect_regressions

    path_a, path_b = Path(run_a), Path(run_b)

    if not path_a.exists() or not path_b.exists():
        console.print("[red]One or both result files not found.[/red]")
        raise typer.Exit(1)

    report_a = load_report(path_a)
    report_b = load_report(path_b)

    console.print(
        f"[bold]Run A:[/bold] {report_a.meta.run_id} ({report_a.passed}/{report_a.total})"
    )
    console.print(
        f"[bold]Run B:[/bold] {report_b.meta.run_id} ({report_b.passed}/{report_b.total})"
    )

    regressions = detect_regressions(report_a, report_b)
    if regressions:
        console.print(f"\n[red]Regressions ({len(regressions)}):[/red]")
        for tid in regressions:
            console.print(f"  [red]- {tid}[/red]")
    else:
        console.print("\n[green]No regressions detected.[/green]")

    delta = report_b.pass_rate - report_a.pass_rate
    color = "green" if delta >= 0 else "red"
    console.print(
        f"\nPass rate: {report_a.pass_rate * 100:.0f}% → "
        f"{report_b.pass_rate * 100:.0f}% ([{color}]{delta * 100:+.0f}%[/{color}])"
    )


def _load_bench_config():
    """Load NornConfig with env overrides applied (bench section drives runs)."""
    from norn.core.config import NornConfig

    cfg = NornConfig.load()
    cfg.apply_env_overrides()
    return cfg


def _build_config_overrides(cfg, model: str) -> dict:
    """Build NORN_CONFIG_OVERRIDES for bench subprocesses from BenchConfig.

    `--model` accepts "provider/model" or a bare model name and actually
    drives the router tiers (it previously wrote a dead `llm` override).
    """
    bench_provider = cfg.bench.provider
    bench_model = cfg.bench.model
    if model:
        if "/" in model:
            bench_provider, _, bench_model = model.partition("/")
        else:
            bench_model = model
    tier = {"provider": bench_provider, "model": bench_model, "api_base": None}
    # 'yolo' mode: all tools auto-approved (subprocess has no TTY for prompts).
    return {
        "permissions": {"mode": "yolo"},
        "agent": {"max_tool_rounds": cfg.bench.max_tool_rounds},
        "router": {
            "enabled": True,
            "domain_routing": True,
            "tiers": {"fast": tier, "standard": tier, "powerful": tier},
        },
    }


def _collect_session_metrics(cfg, result) -> None:
    """Best-effort: attach tool/LLM counts from the task's JSONL session log.

    The `norn run` subprocess persists its session id to
    ~/.norn/state/last_session; we match its events in today's log file.
    Empty under --docker (logs live inside the container) — acceptable.
    """
    from datetime import UTC, datetime

    from benchmarks.runner.metrics import count_by_tool, count_events, load_session_events

    try:
        sid = (Path.home() / ".norn" / "state" / "last_session").read_text().strip()
        log_dir = Path(cfg.logging.file_dir).expanduser()
        log_path = log_dir / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        events = load_session_events(log_path, sid)
        result.tool_counts = count_by_tool(events)
        result.llm_calls = count_events(events, "llm.complete")
    except Exception:
        pass


def _run_tasks(
    tasks: list,
    *,
    config_overrides: dict,
    meta,
    delay_seconds: float,
    n_runs: int = 1,
    docker: bool = False,
    image: str = "",
    post_result=None,
):
    """Run tasks sequentially and return a RunReport.

    Shared by `bench run` and `bench guard`. ``post_result(task, result)``
    is an optional enrichment hook (judge scoring, session metrics).
    """
    import asyncio

    from benchmarks.runner.evaluator import evaluate_task
    from benchmarks.runner.executor import execute_task
    from benchmarks.runner.models import RunReport
    from benchmarks.runner.sandbox import cleanup_sandbox, create_sandbox

    if docker:
        from benchmarks.runner.docker_sandbox import execute_task_docker

    results = []
    first = True
    for i, task in enumerate(tasks, 1):
        runs = max(task.n_runs, n_runs)
        for run_idx in range(runs):
            # Brief pause between executions (provider rate-limit hygiene).
            if not first:
                import time as _time

                _time.sleep(delay_seconds)
            first = False
            label = f"[{i}/{len(tasks)}] {task.id}"
            if runs > 1:
                label += f" (run {run_idx + 1}/{runs})"
            console.print(f"{label}...", end=" ")
            sandbox = create_sandbox(task)
            try:
                if docker:
                    trace = asyncio.run(
                        execute_task_docker(
                            task,
                            sandbox_dir=sandbox,
                            config_overrides=config_overrides,
                            image=image,
                        )
                    )
                else:
                    trace = asyncio.run(
                        execute_task(task, sandbox_dir=sandbox, config_overrides=config_overrides)
                    )
                result = evaluate_task(task, trace, sandbox_dir=sandbox)
                if post_result is not None:
                    post_result(task, result)
                results.append(result)

                status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
                console.print(f"{status} ({result.latency_ms}ms)")
            except Exception as e:
                console.print(f"[red]ERROR: {e}[/red]")
            finally:
                cleanup_sandbox(sandbox)

    return RunReport(meta=meta, results=results)


def _make_run_meta(model_label: str):
    import subprocess
    import uuid
    from datetime import UTC, datetime

    from benchmarks.runner.models import RunMeta

    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    return RunMeta(
        run_id=f"{git_sha}-{uuid.uuid4().hex[:8]}",
        model=model_label,
        agent_version="0.1.0",
        timestamp=datetime.now(UTC),
    )


@bench_app.command("run")
def bench_run(
    tasks_dir: TasksDirOption = "",
    results_dir: ResultsDirOption = "",
    category: str = typer.Option("", help="Filter by category"),
    task_id: str = typer.Option("", help="Run single task by ID"),
    model: str = typer.Option("", help="Override bench model ('provider/model' or model name)"),
    n_runs: int = typer.Option(0, "--n-runs", help="Runs per task (0 = bench.default_n_runs)"),
    judge: bool = typer.Option(
        False, "--judge", help="Score each task with LLM-as-judge (uses bench.judge config)"
    ),
    judge_model: str = typer.Option(
        "", "--judge-model", help="Override judge model (provider/model)"
    ),
    docker: bool = typer.Option(
        False, "--docker", help="Run each task in an isolated Docker container"
    ),
    docker_image: str = typer.Option(
        "", "--docker-image", help="Docker image to use (default: norn-bench:latest)"
    ),
    build: bool = typer.Option(
        False, "--build", help="Build/rebuild the Docker image before running (requires --docker)"
    ),
) -> None:
    """Run benchmark tasks against Norn."""
    _ensure_benchmarks_importable()
    import asyncio

    from benchmarks.runner.loader import load_tasks
    from benchmarks.runner.persistence import save_report

    cfg = _load_bench_config()

    image = ""
    if docker:
        from benchmarks.runner.docker_sandbox import (
            DEFAULT_IMAGE,
            build_image,
            image_exists,
        )

        image = docker_image or DEFAULT_IMAGE
        if build or not image_exists(image):
            console.print(f"[bold]Building Docker image {image}...[/bold]")
            build_image(image=image)

    td = Path(tasks_dir) if tasks_dir else _default_tasks_dir()
    rd = Path(results_dir) if results_dir else _default_results_dir()
    cat_filter = category if category else None
    tid_filter = task_id if task_id else None

    tasks = load_tasks(tasks_dir=td, category=cat_filter, task_id=tid_filter)
    if not tasks:
        console.print("[dim]No tasks to run.[/dim]")
        return

    if cfg.bench.parallel_tasks > 1:
        console.print(
            "[yellow]bench.parallel_tasks > 1 is not supported yet; running sequentially.[/yellow]"
        )

    config_overrides = _build_config_overrides(cfg, model)
    tier = config_overrides["router"]["tiers"]["fast"]
    model_label = f"{tier['provider']}/{tier['model']}"
    meta = _make_run_meta(model_label)

    # Judge: opt-in via --judge (never a surprise LLM bill). judge_task
    # returns None on any failure, so the column degrades to '-'.
    judge_llm = None
    if judge:
        from norn.core.router import build_litellm_provider

        jm = judge_model or cfg.bench.judge.model
        j_provider, _, j_model = jm.partition("/")
        judge_llm = build_litellm_provider(j_provider, j_model, None)

    def _post_result(task, result) -> None:
        _collect_session_metrics(cfg, result)
        if judge_llm is not None:
            from benchmarks.runner.judge import judge_task

            result.judge_score = asyncio.run(
                judge_task(task, result.trace.stdout[-8000:], llm=judge_llm)
            )

    effective_n_runs = n_runs or cfg.bench.default_n_runs
    console.print(f"[bold]Running {len(tasks)} benchmark tasks ({model_label})...[/bold]\n")
    report = _run_tasks(
        tasks,
        config_overrides=config_overrides,
        meta=meta,
        delay_seconds=cfg.bench.inter_task_delay_seconds,
        n_runs=effective_n_runs,
        docker=docker,
        image=image,
        post_result=_post_result,
    )

    filepath = save_report(report, results_dir=rd)
    console.print(
        f"\n[bold]Results:[/bold] {report.passed}/{report.total} passed "
        f"({report.pass_rate * 100:.0f}%)"
    )
    judged = [r for r in report.results if r.judge_score]
    if judged:
        avg = sum(r.judge_score.average() for r in judged) / len(judged)
        console.print(f"[bold]Judge avg:[/bold] {avg:.1f}/10 over {len(judged)} tasks")
    console.print(f"[dim]Saved to {filepath}[/dim]")


@bench_app.command("guard")
def bench_guard(
    tasks_dir: TasksDirOption = "",
    results_dir: ResultsDirOption = "",
    max_tasks: int = typer.Option(5, help="Max easy tasks to run"),
    update_baseline: bool = typer.Option(
        False, "--update-baseline", help="Record this run as the new guard baseline"
    ),
    revert_lessons: bool = typer.Option(
        False,
        "--revert-lessons",
        help="Restore lessons.md from its backup if a regression is detected",
    ),
) -> None:
    """W3.3 self-improvement safety net: mini-bench vs a stored baseline.

    Runs up to `max_tasks` easy tasks and compares against the last guard
    baseline (results/guard/). Exit code 1 on regression (CI-friendly).
    Run after `/reflect` persists a new lesson; `--revert-lessons` undoes
    the lesson when it regresses.
    """
    _ensure_benchmarks_importable()
    from benchmarks.runner.loader import load_tasks
    from benchmarks.runner.persistence import list_reports, load_report, save_report
    from benchmarks.runner.reporter import detect_regressions

    cfg = _load_bench_config()
    td = Path(tasks_dir) if tasks_dir else _default_tasks_dir()
    rd = (Path(results_dir) if results_dir else _default_results_dir()) / "guard"

    tasks = load_tasks(tasks_dir=td, difficulty="easy")[:max_tasks]
    if not tasks:
        console.print("[dim]No easy tasks found for the guard suite.[/dim]")
        return

    config_overrides = _build_config_overrides(cfg, "")
    tier = config_overrides["router"]["tiers"]["fast"]
    meta = _make_run_meta(f"{tier['provider']}/{tier['model']}")

    console.print(f"[bold]Guard: running {len(tasks)} easy tasks...[/bold]\n")
    report = _run_tasks(
        tasks,
        config_overrides=config_overrides,
        meta=meta,
        delay_seconds=cfg.bench.inter_task_delay_seconds,
    )

    baselines = list_reports(results_dir=rd)
    if update_baseline or not baselines:
        filepath = save_report(report, results_dir=rd)
        console.print(
            f"\n[green]Guard baseline recorded:[/green] {report.passed}/{report.total} "
            f"([dim]{filepath}[/dim])"
        )
        return

    baseline = load_report(baselines[-1])
    regressions = detect_regressions(baseline, report)
    delta = report.pass_rate - baseline.pass_rate
    color = "green" if delta >= 0 else "red"
    console.print(
        f"\nPass rate: {baseline.pass_rate * 100:.0f}% → "
        f"{report.pass_rate * 100:.0f}% ([{color}]{delta * 100:+.0f}%[/{color}])"
    )

    if not regressions:
        console.print("[green]No regressions — lessons are safe.[/green]")
        return

    console.print(f"[red]Regressions ({len(regressions)}):[/red]")
    for tid in regressions:
        console.print(f"  [red]- {tid}[/red]")
    if revert_lessons:
        from norn.cli.main import _build_memory_store

        store = _build_memory_store(cfg)
        if store is not None and store.restore_lessons_backup():
            console.print("[yellow]lessons.md restored from backup.[/yellow]")
        else:
            console.print("[yellow]No lessons backup to restore.[/yellow]")
    raise typer.Exit(1)
