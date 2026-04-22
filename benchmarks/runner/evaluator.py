"""Task evaluator: runs eval_command in sandbox and builds TaskResult."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from benchmarks.runner.models import ExecutionTrace, TaskDef, TaskResult


@dataclass
class EvalCommandResult:
    """Result of running the eval_command in a sandbox."""

    returncode: int
    stdout: str
    stderr: str


def run_eval_command(
    command: str,
    *,
    cwd: Path,
    timeout: int = 60,
) -> EvalCommandResult:
    """Run the eval_command (e.g. `pytest tests/`) in the sandbox.

    Returns an EvalCommandResult with returncode, stdout, stderr.
    """
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return EvalCommandResult(
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )
    except subprocess.TimeoutExpired:
        return EvalCommandResult(
            returncode=-1,
            stdout="",
            stderr="eval_command timed out",
        )


def evaluate_task(
    task: TaskDef,
    trace: ExecutionTrace,
    *,
    sandbox_dir: Path,
) -> TaskResult:
    """Run eval_command in the sandbox and build a TaskResult.

    If the trace is timed out, returns a timeout result without running eval.

    Args:
        task: The task definition (contains eval_command).
        trace: The execution trace from the executor.
        sandbox_dir: Path to the sandbox working directory.

    Returns:
        TaskResult with binary_pass, success, eval output, and latency.
    """
    if trace.timed_out:
        return TaskResult.timeout(task, trace)

    eval_result = run_eval_command(task.eval_command, cwd=sandbox_dir)
    binary_pass = eval_result.returncode == 0

    return TaskResult(
        task_id=task.id,
        success=binary_pass,
        timed_out=False,
        latency_ms=trace.duration_ms,
        trace=trace,
        binary_pass=binary_pass,
        eval_returncode=eval_result.returncode,
        eval_stdout=eval_result.stdout,
    )
