"""Tests for benchmark reporter (Rich + markdown)."""

from __future__ import annotations

from datetime import datetime, timezone
from io import StringIO

from benchmarks.runner.models import RunMeta, RunReport, TaskResult, JudgeScore
from benchmarks.runner.reporter import (
    render_summary_table,
    render_markdown_report,
    detect_regressions,
)


def _make_report(results: list[TaskResult] | None = None) -> RunReport:
    meta = RunMeta(
        run_id="test-run",
        model="test-model",
        agent_version="0.1.0",
        timestamp=datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc),
    )
    if results is None:
        results = [
            TaskResult(task_id="code-gen-001", success=True, latency_ms=1500, eval_returncode=0),
            TaskResult(task_id="code-gen-002", success=False, latency_ms=3000, eval_returncode=1),
            TaskResult(task_id="bug-fix-001", success=True, latency_ms=2000, eval_returncode=0),
        ]
    return RunReport(meta=meta, results=results)


def test_render_summary_table_no_error():
    """render_summary_table runs without error."""
    report = _make_report()
    # Should not raise
    output = render_summary_table(report)
    assert output is not None


def test_render_summary_table_contains_pass_rate():
    """Summary contains pass rate information."""
    report = _make_report()
    output = render_summary_table(report)
    # 2/3 passed → should contain something about pass rate
    assert "2" in output or "66" in output or "pass" in output.lower()


def test_render_markdown_report():
    """Markdown report is a string containing key sections."""
    report = _make_report()
    md = render_markdown_report(report)
    assert isinstance(md, str)
    assert "# " in md or "## " in md  # has headings
    assert "test-model" in md
    assert "code-gen-001" in md


def test_render_markdown_report_empty_results():
    """Markdown report handles empty results gracefully."""
    report = RunReport(meta=RunMeta(run_id="empty", model="m"))
    md = render_markdown_report(report)
    assert isinstance(md, str)
    assert "0" in md or "empty" in md.lower() or "no" in md.lower()


def test_detect_regressions_finds_regression():
    """detect_regressions finds tasks that went from pass to fail."""
    baseline = _make_report(
        [
            TaskResult(task_id="t1", success=True),
            TaskResult(task_id="t2", success=True),
        ]
    )
    current = _make_report(
        [
            TaskResult(task_id="t1", success=True),
            TaskResult(task_id="t2", success=False),  # regression!
        ]
    )
    regressions = detect_regressions(baseline, current)
    assert len(regressions) == 1
    assert regressions[0] == "t2"


def test_detect_regressions_no_regression():
    """No regressions when all still pass."""
    baseline = _make_report(
        [
            TaskResult(task_id="t1", success=True),
        ]
    )
    current = _make_report(
        [
            TaskResult(task_id="t1", success=True),
        ]
    )
    assert detect_regressions(baseline, current) == []


def test_detect_regressions_new_task_not_regression():
    """A new task (not in baseline) is not a regression."""
    baseline = _make_report(
        [
            TaskResult(task_id="t1", success=True),
        ]
    )
    current = _make_report(
        [
            TaskResult(task_id="t1", success=True),
            TaskResult(task_id="t2", success=False),  # new task, not regression
        ]
    )
    assert detect_regressions(baseline, current) == []
