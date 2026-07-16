"""Tests for Docker-based benchmark execution (Phase 6a)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.runner.docker_sandbox import (
    DEFAULT_IMAGE,
    build_image,
    execute_task_docker,
    image_exists,
)
from benchmarks.runner.models import TaskDef


def _make_task(**kwargs) -> TaskDef:
    defaults = dict(
        id="test-001",
        category="code-gen",
        description="test task",
        prompt="Write hello world",
        seed_dir="seed/",
        eval_command="echo ok",
        timeout_seconds=30,
    )
    defaults.update(kwargs)
    return TaskDef(**defaults)


class TestImageExists:
    def test_returns_true_when_inspect_succeeds(self):
        mock_result = MagicMock(returncode=0)
        with patch("subprocess.run", return_value=mock_result) as mock_run:
            assert image_exists("norn-bench:latest") is True
        mock_run.assert_called_once()
        assert "docker" in mock_run.call_args[0][0]

    def test_returns_false_when_inspect_fails(self):
        mock_result = MagicMock(returncode=1)
        with patch("subprocess.run", return_value=mock_result):
            assert image_exists("norn-bench:latest") is False


class TestBuildImage:
    def test_invokes_docker_build(self, tmp_path):
        dockerfile = tmp_path / "benchmarks" / "docker" / "Dockerfile"
        dockerfile.parent.mkdir(parents=True)
        dockerfile.write_text("FROM python:3.12-slim\n")

        with patch("subprocess.run") as mock_run:
            build_image(image="norn-bench:test", dockerfile=dockerfile)

        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert "docker" in args
        assert "build" in args
        assert "norn-bench:test" in args

    def test_raises_on_build_failure(self, tmp_path):
        import subprocess

        dockerfile = tmp_path / "Dockerfile"
        dockerfile.write_text("FROM scratch\n")

        with (
            patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "docker")),
            pytest.raises(subprocess.CalledProcessError),
        ):
            build_image(image="x", dockerfile=dockerfile)


class TestExecuteTaskDocker:
    @pytest.mark.asyncio
    async def test_success_returns_trace(self, tmp_path):
        task = _make_task()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"Hello world\n", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            trace = await execute_task_docker(task, sandbox_dir=tmp_path)

        assert trace.exit_code == 0
        assert "Hello world" in trace.stdout
        assert trace.timed_out is False
        call_args = mock_exec.call_args[0]
        assert "docker" in call_args
        assert "run" in call_args

    @pytest.mark.asyncio
    async def test_mounts_sandbox_dir(self, tmp_path):
        task = _make_task()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await execute_task_docker(task, sandbox_dir=tmp_path)

        call_args = mock_exec.call_args[0]
        mount_arg = f"{tmp_path}:/workspace"
        assert mount_arg in call_args

    @pytest.mark.asyncio
    async def test_passes_config_overrides_as_env(self, tmp_path):
        task = _make_task()
        config_overrides = {"permissions": {"mode": "yolo"}}
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await execute_task_docker(task, sandbox_dir=tmp_path, config_overrides=config_overrides)

        call_args = mock_exec.call_args[0]
        env_pairs = [call_args[i + 1] for i, a in enumerate(call_args) if a == "-e"]
        override_pair = next(p for p in env_pairs if p.startswith("NORN_CONFIG_OVERRIDES="))
        parsed = json.loads(override_pair.split("=", 1)[1])
        assert parsed["permissions"]["mode"] == "yolo"

    @pytest.mark.asyncio
    async def test_uses_default_image_when_unset(self, tmp_path):
        task = _make_task()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await execute_task_docker(task, sandbox_dir=tmp_path)

        call_args = mock_exec.call_args[0]
        assert DEFAULT_IMAGE in call_args

    @pytest.mark.asyncio
    async def test_timeout_kills_named_container(self, tmp_path):
        task = _make_task(timeout_seconds=1)

        run_process = AsyncMock()
        run_process.communicate = AsyncMock(side_effect=TimeoutError())

        kill_process = AsyncMock()
        kill_process.wait = AsyncMock()

        with patch(
            "asyncio.create_subprocess_exec",
            side_effect=[run_process, kill_process],
        ) as mock_exec:
            trace = await execute_task_docker(task, sandbox_dir=tmp_path)

        assert trace.timed_out is True
        # Second call should be `docker kill <container_name>`
        kill_call_args = mock_exec.call_args_list[1][0]
        assert kill_call_args[0] == "docker"
        assert kill_call_args[1] == "kill"

    @pytest.mark.asyncio
    async def test_forwards_llm_credential_env_vars(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-123")
        task = _make_task()
        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(return_value=(b"", b""))
        mock_process.returncode = 0

        with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
            await execute_task_docker(task, sandbox_dir=tmp_path)

        call_args = mock_exec.call_args[0]
        env_pairs = [call_args[i + 1] for i, a in enumerate(call_args) if a == "-e"]
        assert "ANTHROPIC_API_KEY=sk-test-123" in env_pairs
