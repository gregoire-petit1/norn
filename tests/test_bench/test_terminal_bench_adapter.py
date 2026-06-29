"""Tests for the terminal-bench adapter."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from benchmarks.adapters.terminal_bench import TerminalBenchTask, _run_norn


class TestTerminalBenchTask:
    def test_required_fields(self):
        task = TerminalBenchTask({"id": "t01", "prompt": "do thing", "working_dir": "/tmp"})
        assert task.id == "t01"
        assert task.prompt == "do thing"

    def test_defaults(self):
        task = TerminalBenchTask({"id": "t01", "prompt": "do thing"})
        assert task.working_dir == "."
        assert task.timeout_seconds == 300


class TestNornRunner:
    @pytest.mark.asyncio
    async def test_run_norn_success(self, tmp_path):
        """_run_norn returns (0, stdout, stderr) on success."""
        mock_proc = AsyncMock()
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"done\n", b""))

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            rc, stdout, stderr = await _run_norn("fix the bug", str(tmp_path), 30)

        assert rc == 0
        assert "done" in stdout

    @pytest.mark.asyncio
    async def test_run_norn_timeout(self, tmp_path):
        """_run_norn returns (1, '', timeout message) on timeout."""
        import asyncio

        mock_proc = AsyncMock()
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()  # sync call — not awaited
        mock_proc.wait = AsyncMock()

        async def _slow_communicate():
            await asyncio.sleep(10)
            return b"", b""

        mock_proc.communicate = _slow_communicate

        with patch("asyncio.create_subprocess_exec", return_value=mock_proc):
            rc, stdout, stderr = await _run_norn("do thing", str(tmp_path), timeout_seconds=1)

        assert rc == 1
        assert "timeout" in stderr.lower() or "Timeout" in stderr


class TestMainCLI:
    def test_parse_inline_json(self, tmp_path, monkeypatch, capsys):
        """--task-json flag parses task and calls solve."""
        from unittest.mock import patch as mp

        task_data = {
            "id": "t01",
            "prompt": "write hello world",
            "working_dir": str(tmp_path),
            "timeout_seconds": 10,
        }

        with mp("benchmarks.adapters.terminal_bench.solve", return_value=0):
            import sys
            from benchmarks.adapters.terminal_bench import main

            monkeypatch.setattr(
                sys, "argv",
                ["tb_adapter", "--task-json", json.dumps(task_data)],
            )
            with pytest.raises(SystemExit) as exc:
                main()
            assert exc.value.code == 0

    def test_task_dir_resolves_relative_working_dir(self, tmp_path, monkeypatch):
        """working_dir relative to --task-dir is resolved to absolute."""
        from unittest.mock import patch as mp

        task_data = {
            "id": "t02",
            "prompt": "do something",
            "working_dir": "workspace",
            "timeout_seconds": 10,
        }

        captured: list[TerminalBenchTask] = []

        def _capture_solve(task: TerminalBenchTask) -> int:
            captured.append(task)
            return 0

        import sys
        from benchmarks.adapters.terminal_bench import main

        monkeypatch.setattr(
            sys, "argv",
            ["tb_adapter", "--task-dir", str(tmp_path), "--task-json", json.dumps(task_data)],
        )
        with mp("benchmarks.adapters.terminal_bench.solve", side_effect=_capture_solve):
            with pytest.raises(SystemExit):
                main()

        assert captured[0].working_dir == str(tmp_path / "workspace")
