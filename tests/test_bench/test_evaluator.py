"""Tests for benchmark task evaluator."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from benchmarks.runner.evaluator import evaluate_task, run_eval_command
from benchmarks.runner.models import ExecutionTrace, TaskDef, TaskResult


def _make_task(**kwargs) -> TaskDef:
    defaults = dict(
        id="test-001",
        category="code-gen",
        description="test task",
        prompt="Write hello world",
        seed_dir="seed/",
        eval_command="uv run pytest tests/ -v",
        timeout_seconds=30,
    )
    defaults.update(kwargs)
    return TaskDef(**defaults)


def _make_trace(**kwargs) -> ExecutionTrace:
    defaults = dict(
        exit_code=0,
        stdout="Hello world\n",
        stderr="",
        duration_ms=1500,
        timed_out=False,
    )
    defaults.update(kwargs)
    return ExecutionTrace(**defaults)


# --- run_eval_command tests ---


def test_run_eval_command_pass(tmp_path):
    """Passing eval_command returns returncode=0 and stdout."""
    result = run_eval_command("echo ok", cwd=tmp_path)
    assert result.returncode == 0
    assert "ok" in result.stdout


def test_run_eval_command_fail(tmp_path):
    """Failing eval_command returns non-zero returncode."""
    result = run_eval_command("false", cwd=tmp_path)
    assert result.returncode != 0


def test_run_eval_command_captures_stderr(tmp_path):
    """stderr is captured from eval_command."""
    result = run_eval_command("echo error >&2", cwd=tmp_path)
    assert "error" in result.stderr


# --- evaluate_task tests ---


def test_evaluate_task_passing_eval(tmp_path):
    """Passing eval_command produces binary_pass=True, success=True."""
    task = _make_task(eval_command="echo ok")
    trace = _make_trace()

    result = evaluate_task(task, trace, sandbox_dir=tmp_path)

    assert result.task_id == "test-001"
    assert result.binary_pass is True
    assert result.success is True
    assert result.eval_returncode == 0
    assert "ok" in result.eval_stdout
    assert result.timed_out is False
    assert result.latency_ms == 1500


def test_evaluate_task_failing_eval(tmp_path):
    """Failing eval_command produces binary_pass=False, success=False."""
    task = _make_task(eval_command="false")
    trace = _make_trace()

    result = evaluate_task(task, trace, sandbox_dir=tmp_path)

    assert result.binary_pass is False
    assert result.success is False
    assert result.eval_returncode != 0


def test_evaluate_task_timed_out_trace(tmp_path):
    """Timed-out trace returns TaskResult.timeout()."""
    task = _make_task()
    trace = ExecutionTrace.timeout(30)

    result = evaluate_task(task, trace, sandbox_dir=tmp_path)

    assert result.timed_out is True
    assert result.success is False
    assert result.latency_ms == 30_000


def test_evaluate_task_extracts_latency(tmp_path):
    """Latency from trace is propagated to TaskResult."""
    task = _make_task(eval_command="echo ok")
    trace = _make_trace(duration_ms=4200)

    result = evaluate_task(task, trace, sandbox_dir=tmp_path)

    assert result.latency_ms == 4200


def test_evaluate_task_captures_eval_output(tmp_path):
    """eval_stdout is populated from eval_command output."""
    task = _make_task(eval_command="echo 'all tests passed'")
    trace = _make_trace()

    result = evaluate_task(task, trace, sandbox_dir=tmp_path)

    assert "all tests passed" in result.eval_stdout
