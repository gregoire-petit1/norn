"""Docker-based execution for benchmark tasks (Phase 6a).

Scope: benchmark runner only, not agent production execution. Each task
runs `norn run <prompt>` inside a fresh container from a pinned image with
the host sandbox dir (see ``sandbox.py``) bind-mounted at /workspace — clean
environment per task, no host process/package-cache contamination, and
comparable to how terminal-bench isolates tasks.

The eval step (pytest, etc.) still runs on the host against the bind-mounted
sandbox_dir, unchanged from the non-Docker path, since files written inside
the container are visible on the host through the mount.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks.runner.models import ExecutionTrace, TaskDef

DEFAULT_IMAGE = "norn-bench:latest"
_DOCKERFILE_PATH = Path(__file__).resolve().parent.parent / "docker" / "Dockerfile"

# LLM provider credentials forwarded into the container.
_FORWARD_ENV_VARS = [
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "OLLAMA_HOST",
]


def image_exists(image: str = DEFAULT_IMAGE) -> bool:
    """Check whether the benchmark Docker image has already been built."""
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


def build_image(image: str = DEFAULT_IMAGE, dockerfile: Path = _DOCKERFILE_PATH) -> None:
    """Build the benchmark Docker image from ``benchmarks/docker/Dockerfile``.

    Raises ``subprocess.CalledProcessError`` on build failure.
    """
    project_root = dockerfile.parent.parent.parent
    subprocess.run(
        ["docker", "build", "-f", str(dockerfile), "-t", image, str(project_root)],
        check=True,
    )


async def execute_task_docker(
    task: TaskDef,
    *,
    sandbox_dir: Path,
    config_overrides: dict[str, Any] | None = None,
    image: str = DEFAULT_IMAGE,
) -> ExecutionTrace:
    """Run ``norn run <prompt>`` inside a fresh, named container for this task."""
    container_name = f"norn-bench-{task.id.replace('/', '-')}-{uuid.uuid4().hex[:8]}"

    docker_args = [
        "docker", "run", "--rm",
        "--name", container_name,
        "-v", f"{sandbox_dir}:/workspace",
        "-w", "/workspace",
    ]

    litellm_config_dir = Path.home() / ".config" / "litellm"
    if litellm_config_dir.is_dir():
        docker_args += ["-v", f"{litellm_config_dir}:/root/.config/litellm"]

    for var in _FORWARD_ENV_VARS:
        value = os.environ.get(var)
        if value:
            docker_args += ["-e", f"{var}={value}"]

    if config_overrides:
        docker_args += ["-e", f"NORN_CONFIG_OVERRIDES={json.dumps(config_overrides)}"]

    docker_args += [image, "norn", "run", task.prompt]

    start = time.monotonic()
    try:
        proc = await asyncio.create_subprocess_exec(
            *docker_args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
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
        # `docker run` (no -d) doesn't propagate host SIGKILL to the
        # container process — the container must be killed by name.
        kill_proc = await asyncio.create_subprocess_exec(
            "docker", "kill", container_name,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await kill_proc.wait()
        return ExecutionTrace(timed_out=True, duration_ms=elapsed_ms)
