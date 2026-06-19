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


@bench_app.command("run")
def bench_run(
    tasks_dir: TasksDirOption = "",
    results_dir: ResultsDirOption = "",
    category: str = typer.Option("", help="Filter by category"),
    task_id: str = typer.Option("", help="Run single task by ID"),
    model: str = typer.Option("", help="Override LLM model"),
) -> None:
    """Run benchmark tasks against Norn."""
    _ensure_benchmarks_importable()
    import asyncio
    import subprocess
    import uuid
    from datetime import datetime, timezone

    from benchmarks.runner.evaluator import evaluate_task
    from benchmarks.runner.executor import execute_task
    from benchmarks.runner.loader import load_tasks
    from benchmarks.runner.models import RunMeta, RunReport
    from benchmarks.runner.persistence import save_report
    from benchmarks.runner.sandbox import cleanup_sandbox, create_sandbox

    td = Path(tasks_dir) if tasks_dir else _default_tasks_dir()
    rd = Path(results_dir) if results_dir else _default_results_dir()
    cat_filter = category if category else None
    tid_filter = task_id if task_id else None

    tasks = load_tasks(tasks_dir=td, category=cat_filter, task_id=tid_filter)
    if not tasks:
        console.print("[dim]No tasks to run.[/dim]")
        return

    # Get git SHA
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        git_sha = "unknown"

    # Use 'yolo' mode: all tools auto-approved (subprocess has no TTY for prompts)
    # Limit tool rounds to 20 — enough headroom for multi-file tasks (write
    # impl + tests + run + iterate) while still capping echo loops.
    config_overrides: dict = {
        "permissions": {"mode": "yolo"},
        "agent": {"max_tool_rounds": 20},
        # Enable router so Ollama session-limit errors auto-fall back to Groq.
        "router": {
            "enabled": True,
            "domain_routing": True,
            "tiers": {
                "fast": {"provider": "ollama", "model": "qwen3-coder:480b-cloud", "api_base": None},
                "standard": {"provider": "groq", "model": "llama-3.3-70b-versatile", "api_base": None},
                "powerful": {"provider": "groq", "model": "llama-3.3-70b-versatile", "api_base": None},
            },
        },
    }
    if model:
        config_overrides["llm"] = {"model": model}

    meta = RunMeta(
        run_id=f"{git_sha}-{uuid.uuid4().hex[:8]}",
        model=model or "default",
        agent_version="0.1.0",
        timestamp=datetime.now(timezone.utc),
    )

    results = []
    console.print(f"[bold]Running {len(tasks)} benchmark tasks...[/bold]\n")

    for i, task in enumerate(tasks, 1):
        # Brief pause between tasks so Groq/Ollama TPM windows partially reset.
        if i > 1:
            import time as _time
            _time.sleep(10)
        console.print(f"[{i}/{len(tasks)}] {task.id}...", end=" ")
        sandbox = create_sandbox(task)
        try:
            trace = asyncio.run(
                execute_task(task, sandbox_dir=sandbox, config_overrides=config_overrides)
            )
            result = evaluate_task(task, trace, sandbox_dir=sandbox)
            results.append(result)

            status = "[green]PASS[/green]" if result.success else "[red]FAIL[/red]"
            console.print(f"{status} ({result.latency_ms}ms)")
        except Exception as e:
            console.print(f"[red]ERROR: {e}[/red]")
        finally:
            cleanup_sandbox(sandbox)

    report = RunReport(meta=meta, results=results)
    filepath = save_report(report, results_dir=rd)
    console.print(
        f"\n[bold]Results:[/bold] {report.passed}/{report.total} passed "
        f"({report.pass_rate * 100:.0f}%)"
    )
    console.print(f"[dim]Saved to {filepath}[/dim]")
