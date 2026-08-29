"""Harbor agent adapter for Norn (harbor-framework/harbor, terminal-bench 2.x).

Implements the ``BaseInstalledAgent`` interface: harbor spins up a task
container, ``install()`` puts Norn inside it, ``run()`` invokes
``norn run <instruction>`` as the in-container agent user.

Usage (from an isolated harbor venv, NOT the Norn project venv — harbor's own
dependency set conflicts with Norn's pinned versions):

    harbor run -d terminal-bench/terminal-bench-2 \\
        -a benchmarks.harbor_adapter.norn_agent:NornAgent \\
        -k <single-task-id>   # smoke test: one task, not the full suite

Auth: reuses the host's cached GitHub Copilot Enterprise OAuth token
(``~/.config/litellm/github_copilot/api-key.json``) by uploading it into the
container, so no raw provider API key is required — same mechanism as
``benchmarks/runner/docker_sandbox.py``.

Scope note: only tested against Debian/Ubuntu-based task images (the
``apt-get`` install path). Alpine/busybox-based tasks are not supported by
this adapter.
"""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# Repo root: benchmarks/harbor_adapter/norn_agent.py -> repo root
_NORN_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_LITELLM_CONFIG_DIR = Path.home() / ".config" / "litellm"

# Same router config as benchmarks/runner/bench_run's config_overrides:
# route everything through GitHub Copilot Enterprise (high rate limits, no
# session cap) rather than a raw provider API key.
_CONFIG_OVERRIDES = {
    "permissions": {"mode": "yolo"},
    # Terminal-bench tasks need far more sequential steps than Norn's trivial
    # internal suite: observed failures (chess-best-move, overfull-hbox) were
    # cut off mid-work at the old cap of 20 before writing the deliverable.
    # auto_verify adds up to 2 self-check turns so the agent confirms its
    # output file exists / tests pass before the container tears down.
    "agent": {"max_tool_rounds": 50, "auto_verify": True, "auto_verify_max_rounds": 2},
    # image_read multimodal: terminal-bench has visual tasks (chess from a
    # rendered board, plots, scanned docs) a text-only agent can't solve.
    # claude-sonnet-4.5 (router tiers below) is vision-capable.
    "flags": {"vision_tools": True},
    "router": {
        "enabled": True,
        "domain_routing": True,
        "tiers": {
            "fast": {"provider": "github_copilot", "model": "claude-sonnet-4.5", "api_base": None},
            "standard": {
                "provider": "github_copilot",
                "model": "claude-sonnet-4.5",
                "api_base": None,
            },
            "powerful": {
                "provider": "github_copilot",
                "model": "claude-sonnet-4.5",
                "api_base": None,
            },
        },
    },
}


class NornAgent(BaseInstalledAgent):
    """Runs Norn inside the harbor task container as an installed CLI agent."""

    @staticmethod
    @override
    def name() -> str:
        return "norn"

    @override
    def get_version_command(self) -> str | None:
        return "norn version"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "apt-get update && "
                "apt-get install -y --no-install-recommends "
                "python3 python3-pip python3-venv git"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )

        await environment.upload_dir(_NORN_REPO_ROOT, "/opt/norn-src")

        if _LITELLM_CONFIG_DIR.is_dir():
            await environment.upload_dir(_LITELLM_CONFIG_DIR, "/root/.config/litellm")

        # `pip install -e .` alone re-resolves deps freely from PyPI, which can
        # silently pick up a newer litellm with undeclared soft-dependencies
        # (e.g. a github_copilot token-refresh path that imports fastapi only
        # when the cached token has expired — invisible in a short host session
        # where the token never needed refreshing). Installing from the
        # project's own uv.lock via `uv export` pins exactly what's verified
        # to work on the host.
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            requirements_path = Path(f.name)
        try:
            subprocess.run(
                [
                    "uv", "export", "--no-hashes",
                    "--no-emit-project",
                    "--format", "requirements-txt",
                    "--extra", "mcp_tools",
                    "-o", str(requirements_path),
                ],
                cwd=_NORN_REPO_ROOT,
                check=True,
                capture_output=True,
            )
            await environment.upload_file(requirements_path, "/opt/requirements.txt")
        finally:
            requirements_path.unlink(missing_ok=True)

        # Install into a dedicated venv rather than the system interpreter.
        # Two task images broke the old `pip install --break-system-packages`
        # path: some ship a pip too old to know that flag (`no such option`),
        # and others fail to upgrade Debian-owned packages
        # (`Cannot uninstall urllib3 ... RECORD file not found`). A fresh venv
        # with an upgraded pip sidesteps both. Symlink `norn` onto PATH so
        # `exec_as_agent` and `norn version` resolve it.
        await self.exec_as_root(
            environment,
            command=(
                "python3 -m venv /opt/norn-venv && "
                "/opt/norn-venv/bin/pip install --upgrade pip && "
                "/opt/norn-venv/bin/pip install -r /opt/requirements.txt && "
                "/opt/norn-venv/bin/pip install --no-deps -e /opt/norn-src && "
                "ln -sf /opt/norn-venv/bin/norn /usr/local/bin/norn"
            ),
        )

    @override
    def populate_context_post_run(self, context: AgentContext) -> None:
        # Reward/pass-fail comes from the task's own verifier (pytest, etc.),
        # not from parsing Norn's stdout — nothing to extract here.
        pass

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        escaped_instruction = shlex.quote(instruction)
        env = {"NORN_CONFIG_OVERRIDES": json.dumps(_CONFIG_OVERRIDES)}

        await self.exec_as_agent(
            environment,
            command=f"norn run {escaped_instruction} 2>&1 | tee /logs/agent/norn.txt",
            env=env,
        )
