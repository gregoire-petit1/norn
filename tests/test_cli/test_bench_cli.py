"""Tests for norn bench CLI sub-app."""

from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from norn.cli.bench import bench_app

runner = CliRunner()


def _write_task(base: Path, category: str, task_id: str) -> None:
    """Create a minimal valid task.yaml."""
    task_dir = base / category / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    seed = task_dir / "seed"
    seed.mkdir(exist_ok=True)
    (seed / "main.py").write_text("# seed\n")
    data = {
        "id": f"{category}-{task_id}",
        "category": category,
        "description": f"Test {task_id}",
        "prompt": f"Do {task_id}",
        "seed_dir": "seed/",
        "eval_command": "echo ok",
    }
    (task_dir / "task.yaml").write_text(yaml.dump(data))


def _write_invalid_task(base: Path, category: str, task_id: str) -> None:
    """Create an invalid task.yaml (missing required fields)."""
    task_dir = base / category / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "task.yaml").write_text("id: incomplete\n")


# --- Tests ---


def test_bench_list(tmp_path):
    """bench list renders available tasks."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "bug-fix", "001-off")

    result = runner.invoke(bench_app, ["list", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "code-gen-001-fib" in result.output
    assert "bug-fix-001-off" in result.output


def test_bench_list_empty(tmp_path):
    """bench list shows message when no tasks found."""
    result = runner.invoke(bench_app, ["list", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no task" in result.output.lower() or "0" in result.output


def test_bench_validate_all_valid(tmp_path):
    """bench validate passes when all tasks are valid."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "code-gen", "002-fizz")

    result = runner.invoke(bench_app, ["validate", "--tasks-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "2" in result.output  # 2 valid


def test_bench_validate_catches_invalid(tmp_path):
    """bench validate catches invalid task YAML."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_invalid_task(tmp_path, "code-gen", "002-bad")

    result = runner.invoke(bench_app, ["validate", "--tasks-dir", str(tmp_path)])
    assert (
        "002-bad" in result.output
        or "invalid" in result.output.lower()
        or "error" in result.output.lower()
    )


def test_bench_report_no_results(tmp_path):
    """bench report handles no results gracefully."""
    result = runner.invoke(bench_app, ["report", "--results-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "no" in result.output.lower() or "empty" in result.output.lower()
