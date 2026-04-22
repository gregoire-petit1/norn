"""Tests for benchmark task executor."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.runner.executor import execute_task
from benchmarks.runner.models import TaskDef, ExecutionTrace


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


@pytest.mark.asyncio
async def test_execute_task_success(tmp_path):
    """Successful execution returns trace with stdout and exit_code=0."""
    task = _make_task()

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"Hello world\n", b""))
    mock_process.returncode = 0
    mock_process.pid = 12345

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        trace = await execute_task(task, sandbox_dir=tmp_path)

    assert trace.exit_code == 0
    assert "Hello world" in trace.stdout
    assert trace.timed_out is False
    assert trace.duration_ms >= 0
    # Verify norn run was called with the prompt
    call_args = mock_exec.call_args
    assert "norn" in str(call_args) or "uv" in str(call_args)


@pytest.mark.asyncio
async def test_execute_task_nonzero_exit(tmp_path):
    """Non-zero exit code is captured in trace."""
    task = _make_task()

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"", b"Error occurred\n"))
    mock_process.returncode = 1
    mock_process.pid = 12345

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        trace = await execute_task(task, sandbox_dir=tmp_path)

    assert trace.exit_code == 1
    assert "Error occurred" in trace.stderr
    assert trace.timed_out is False


@pytest.mark.asyncio
async def test_execute_task_timeout(tmp_path):
    """Timeout returns ExecutionTrace with timed_out=True."""
    task = _make_task(timeout_seconds=1)

    mock_process = AsyncMock()
    # Simulate timeout: communicate raises TimeoutError
    mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
    mock_process.kill = MagicMock()
    mock_process.wait = AsyncMock()
    mock_process.pid = 12345
    mock_process.returncode = -9

    with patch("asyncio.create_subprocess_exec", return_value=mock_process):
        trace = await execute_task(task, sandbox_dir=tmp_path)

    assert trace.timed_out is True
    assert trace.duration_ms >= 0


@pytest.mark.asyncio
async def test_execute_task_config_overrides(tmp_path):
    """NORN_CONFIG_OVERRIDES is set in subprocess env."""
    task = _make_task()
    config_overrides = {"permissions": {"mode": "strict"}}

    mock_process = AsyncMock()
    mock_process.communicate = AsyncMock(return_value=(b"ok\n", b""))
    mock_process.returncode = 0
    mock_process.pid = 12345

    with patch("asyncio.create_subprocess_exec", return_value=mock_process) as mock_exec:
        trace = await execute_task(
            task,
            sandbox_dir=tmp_path,
            config_overrides=config_overrides,
        )

    # Verify env was passed with NORN_CONFIG_OVERRIDES
    call_kwargs = mock_exec.call_args
    env = call_kwargs.kwargs.get("env") or call_kwargs[1].get("env")
    assert env is not None
    assert "NORN_CONFIG_OVERRIDES" in env
    parsed = json.loads(env["NORN_CONFIG_OVERRIDES"])
    assert parsed["permissions"]["mode"] == "strict"
