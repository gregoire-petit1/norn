"""Benchmark reporter: Rich table for terminal + markdown for files."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.table import Table

from benchmarks.runner.models import RunReport


def render_summary_table(report: RunReport) -> str:
    """Render a Rich summary table as a string.

    Returns the rendered string (useful for testing and for
    printing to a non-default console).
    """
    table = Table(title="Benchmark Results", show_lines=True)
    table.add_column("Task ID", style="bold")
    table.add_column("Pass", justify="center")
    table.add_column("Latency (ms)", justify="right")
    table.add_column("Eval RC", justify="center")
    table.add_column("Judge Avg", justify="center")

    for r in report.results:
        status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
        judge = f"{r.judge_score.average():.1f}" if r.judge_score else "-"
        table.add_row(
            r.task_id,
            status,
            str(r.latency_ms),
            str(r.eval_returncode),
            judge,
        )

    # Summary row
    table.add_section()
    rate = f"{report.pass_rate * 100:.0f}%"
    avg_latency = (
        str(int(sum(r.latency_ms for r in report.results) / len(report.results)))
        if report.results
        else "0"
    )
    table.add_row(
        f"[bold]TOTAL ({report.total})[/bold]",
        f"[bold]{report.passed}/{report.total} ({rate})[/bold]",
        avg_latency,
        "",
        "",
    )

    buf = StringIO()
    console = Console(file=buf, force_terminal=False, width=120)
    console.print(table)
    return buf.getvalue()


def render_markdown_report(report: RunReport) -> str:
    """Render a markdown report string."""
    lines: list[str] = []
    lines.append(f"# Benchmark Report: {report.meta.run_id}")
    lines.append("")
    lines.append(f"- **Model:** {report.meta.model}")
    lines.append(f"- **Agent version:** {report.meta.agent_version}")
    lines.append(f"- **Timestamp:** {report.meta.timestamp.isoformat()}")
    lines.append(f"- **Total tasks:** {report.total}")
    lines.append(f"- **Passed:** {report.passed}/{report.total} ({report.pass_rate * 100:.0f}%)")
    lines.append("")

    if not report.results:
        lines.append("No results.")
        return "\n".join(lines)

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| Task ID | Pass | Latency (ms) | Eval RC | Judge Avg |")
    lines.append("|---------|------|-------------|---------|-----------|")
    for r in report.results:
        status = "PASS" if r.success else "FAIL"
        judge = f"{r.judge_score.average():.1f}" if r.judge_score else "-"
        lines.append(f"| {r.task_id} | {status} | {r.latency_ms} | {r.eval_returncode} | {judge} |")

    lines.append("")
    metrics_section = render_metrics_section(report)
    if metrics_section:
        lines.append(metrics_section)
    return "\n".join(lines)


def render_metrics_section(report: RunReport) -> str:
    """Render a per-task metrics table (SOTA v2, workstream B).

    Returns "" when no result carries metrics, so reports produced by
    runs without session-metrics extraction are unchanged.
    """
    with_metrics = [r for r in report.results if r.tool_counts or r.llm_calls]
    if not with_metrics:
        return ""
    lines = [
        "## Metrics",
        "",
        "| Task ID | LLM calls | Tool calls |",
        "|---------|-----------|------------|",
    ]
    for r in with_metrics:
        tools = ", ".join(f"{name}×{n}" for name, n in sorted(r.tool_counts.items())) or "-"
        lines.append(f"| {r.task_id} | {r.llm_calls} | {tools} |")
    lines.append("")
    return "\n".join(lines)


def detect_regressions(baseline: RunReport, current: RunReport) -> list[str]:
    """Detect task IDs that regressed (passed in baseline, failed in current).

    New tasks (in current but not in baseline) are NOT considered regressions.
    Returns list of regressed task IDs.
    """
    baseline_pass = {r.task_id for r in baseline.results if r.success}
    regressions: list[str] = []
    for r in current.results:
        if r.task_id in baseline_pass and not r.success:
            regressions.append(r.task_id)
    return regressions
