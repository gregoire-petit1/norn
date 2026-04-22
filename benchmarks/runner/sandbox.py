"""Sandbox creation: isolated tmpdir with git for each benchmark task."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.runner.models import TaskDef


def create_sandbox(task: TaskDef) -> Path:
    """Copy seed/ to a tmpdir, git init + commit, return sandbox path."""
    seed = task.absolute_seed_dir
    tmpdir = Path(tempfile.mkdtemp(prefix=f"norn-bench-{task.id}-"))
    shutil.copytree(seed, tmpdir, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=tmpdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmpdir, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=bench@norn",
            "-c",
            "user.name=Bench",
            "commit",
            "-qm",
            "seed",
        ],
        cwd=tmpdir,
        check=True,
    )
    return tmpdir


def cleanup_sandbox(path: Path) -> None:
    """Remove sandbox directory. No-op if path doesn't exist."""
    shutil.rmtree(path, ignore_errors=True)
