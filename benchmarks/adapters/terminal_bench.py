"""Norn adapter for terminal-bench (harbor-labs/terminal-bench).

Pin terminal-bench dataset version when running:
    tb run --agent benchmarks/adapters/terminal_bench.py --dataset v0.1.1

terminal-bench API contract (v0.1.1):
    tb invokes this script as:
        python terminal_bench.py --task-dir <dir> --task-file task.json

    task.json schema:
        {
            "id": str,
            "prompt": str,
            "working_dir": str,   # absolute path inside container / sandbox
            "timeout_seconds": int
        }

    This script must write to stdout and exit 0 on success.
    Verification is handled externally by tb using the oracle script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path


# --------------------------------------------------------------------------- #
# Task schema
# --------------------------------------------------------------------------- #

class TerminalBenchTask:
    """Minimal representation of a terminal-bench task."""

    def __init__(self, data: dict) -> None:
        self.id: str = data["id"]
        self.prompt: str = data["prompt"]
        self.working_dir: str = data.get("working_dir", ".")
        self.timeout_seconds: int = data.get("timeout_seconds", 300)


# --------------------------------------------------------------------------- #
# Norn runner
# --------------------------------------------------------------------------- #

async def _run_norn(prompt: str, cwd: str, timeout_seconds: int) -> tuple[int, str, str]:
    """Run `norn run <prompt>` in the given directory.

    Returns (returncode, stdout, stderr).
    """
    env = os.environ.copy()
    # Disable interactive mode for headless execution
    env.setdefault("NORN_PERMISSION_MODE", "yolo")

    cmd = [sys.executable, "-m", "norn", "run", prompt]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=env,
        )
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(),
            timeout=timeout_seconds,
        )
        return (
            proc.returncode or 0,
            stdout_bytes.decode(errors="replace"),
            stderr_bytes.decode(errors="replace"),
        )
    except asyncio.TimeoutError:
        try:
            proc.kill()  # type: ignore[possibly-undefined]
            await proc.wait()  # type: ignore[possibly-undefined]
        except OSError:
            pass
        return (1, "", f"Timeout: task exceeded {timeout_seconds}s")


# --------------------------------------------------------------------------- #
# Adapter entry point
# --------------------------------------------------------------------------- #

def solve(task: TerminalBenchTask) -> int:
    """Run Norn on a terminal-bench task. Returns exit code."""
    returncode, stdout, stderr = asyncio.run(
        _run_norn(task.prompt, task.working_dir, task.timeout_seconds)
    )
    if stdout:
        sys.stdout.write(stdout)
    if stderr:
        sys.stderr.write(stderr)
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Norn agent adapter for terminal-bench")
    parser.add_argument("--task-dir", type=Path, help="Directory containing the task")
    parser.add_argument("--task-file", type=Path, help="Path to task JSON file")
    parser.add_argument("--task-json", type=str, help="Task as inline JSON string")
    args = parser.parse_args()

    if args.task_file:
        task_data = json.loads(args.task_file.read_text())
    elif args.task_json:
        task_data = json.loads(args.task_json)
    else:
        # Fallback: read task from stdin (common tb pattern)
        task_data = json.load(sys.stdin)

    # If task specifies a relative working_dir, resolve against --task-dir
    if args.task_dir and not Path(task_data.get("working_dir", ".")).is_absolute():
        task_data["working_dir"] = str(args.task_dir / task_data.get("working_dir", "."))

    task = TerminalBenchTask(task_data)
    sys.exit(solve(task))


if __name__ == "__main__":
    main()
