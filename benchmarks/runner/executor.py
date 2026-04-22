"""Task executor: runs norn against a task in a sandbox directory."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

from benchmarks.runner.models import ExecutionTrace, TaskDef


async def execute_task(
    task: TaskDef,
    *,
    sandbox_dir: Path,
    config_overrides: dict[str, Any] | None = None,
) -> ExecutionTrace:
    """Execute `norn run <prompt>` in the sandbox and return an ExecutionTrace.

    Args:
        task: The task definition.
        sandbox_dir: Path to the sandbox working directory.
        config_overrides: Optional dict merged into NORN_CONFIG_OVERRIDES env var.

    Returns:
        ExecutionTrace with exit_code, stdout, stderr, duration_ms, timed_out.
    """
    env = os.environ.copy()
    if config_overrides:
        env["NORN_CONFIG_OVERRIDES"] = json.dumps(config_overrides)

    cmd = ["uv", "run", "norn", "run", task.prompt]

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(sandbox_dir),
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=task.timeout_seconds,
        )
        elapsed_ms = int((time.monotonic() - start) * 1000)
        return ExecutionTrace(
            exit_code=proc.returncode,
            stdout=stdout_bytes.decode(errors="replace"),
            stderr=stderr_bytes.decode(errors="replace"),
            duration_ms=elapsed_ms,
            timed_out=False,
        )
    except asyncio.TimeoutError:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        # Kill the timed-out process
        with _suppress_os_errors():
            proc.kill()  # type: ignore[possibly-undefined]
            await proc.wait()  # type: ignore[possibly-undefined]
        return ExecutionTrace(
            timed_out=True,
            duration_ms=elapsed_ms,
        )


class _suppress_os_errors:
    """Context manager that suppresses OSError (e.g. process already dead)."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None and issubclass(exc_type, OSError):
            return True
        return False
