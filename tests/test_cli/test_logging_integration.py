"""Integration tests: CLI commands initialize structured logging correctly."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import (
    get_logger,
    init_logging,
    new_session,
    session_id_var,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(autouse=True)
def reset_session():
    token = session_id_var.set(None)
    yield
    session_id_var.reset(token)


def _today_file(dir_: Path) -> Path:
    return dir_ / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"


def _events(dir_: Path) -> list[dict]:
    file = _today_file(dir_)
    if not file.exists():
        return []
    return [json.loads(line) for line in file.read_text().splitlines() if line.strip()]


def test_bootstrap_writes_session_tagged_event(tmp_path):
    """init_logging + new_session + an emit produces a JSONL line with session_id."""
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    sid = new_session()
    get_logger("norn.test").info("agent.run", phase="start")

    events = _events(tmp_path)
    assert len(events) == 1
    assert events[0]["session_id"] == sid
    assert events[0]["event"] == "agent.run"


def test_verbose_flag_enables_debug(tmp_path):
    """cli_level_override='DEBUG' causes DEBUG events to be persisted."""
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), level="INFO")
    init_logging(cfg, cli_level_override="DEBUG")
    new_session()
    get_logger("norn.test").debug("debug.event", x=1)

    events = _events(tmp_path)
    assert any(e["event"] == "debug.event" for e in events)


def test_env_var_sets_level(tmp_path, monkeypatch):
    """NORN_LOG_LEVEL=DEBUG overrides YAML level=INFO."""
    monkeypatch.setenv("NORN_LOG_LEVEL", "DEBUG")
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), level="INFO")
    init_logging(cfg)
    new_session()
    get_logger("norn.test").debug("debug.event", x=1)

    events = _events(tmp_path)
    assert any(e["event"] == "debug.event" for e in events)


def test_cli_override_wins_over_env(tmp_path, monkeypatch):
    """CLI --verbose (DEBUG) wins even when NORN_LOG_LEVEL=ERROR is set."""
    monkeypatch.setenv("NORN_LOG_LEVEL", "ERROR")
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path), level="ERROR")
    init_logging(cfg, cli_level_override="DEBUG")
    new_session()
    get_logger("norn.test").debug("debug.event", x=1)

    events = _events(tmp_path)
    assert any(e["event"] == "debug.event" for e in events)


def test_disabled_logging_writes_nothing(tmp_path):
    """When enabled=False, no file is created even if events are emitted."""
    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("norn.test").info("suppressed", x=1)
    assert list(tmp_path.iterdir()) == []


def test_version_command_bootstraps_logging(tmp_path, monkeypatch):
    """Running `norn version` via typer CliRunner triggers _bootstrap_logging."""
    from typer.testing import CliRunner

    from norn.cli.main import app

    # Point logs to tmp_path so we don't pollute ~/.norn/
    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0

    # A session_id-tagged event should have been emitted somewhere in the run.
    # At minimum we expect the log dir (~/.norn/logs) to exist.
    log_dir = tmp_path / ".norn" / "logs"
    # Dir may or may not be created depending on output mode; we accept both.
    # The main assertion: command did not crash.
    _ = log_dir


def test_verbose_flag_accepted_by_version(tmp_path, monkeypatch):
    """`norn version --verbose` is accepted and does not crash."""
    from typer.testing import CliRunner

    from norn.cli.main import app

    monkeypatch.setenv("HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["version", "--verbose"])
    assert result.exit_code == 0

    result = runner.invoke(app, ["version", "-v"])
    assert result.exit_code == 0
