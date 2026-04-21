"""Tests for console and file sinks."""

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


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _today_file(dir_: Path) -> Path:
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    return dir_ / f"{today}.jsonl"


def test_file_sink_writes_jsonl(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    sid = new_session()
    log = get_logger("test")
    log.info("agent.run", phase="start")

    file = _today_file(tmp_path)
    assert file.exists()
    events = _read_jsonl(file)
    assert len(events) == 1
    assert events[0]["event"] == "agent.run"
    assert events[0]["phase"] == "start"
    assert events[0]["session_id"] == sid


def test_file_sink_multiple_events(tmp_path):
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    log = get_logger("test")
    log.info("e1", a=1)
    log.info("e2", b=2)

    events = _read_jsonl(_today_file(tmp_path))
    names = [e["event"] for e in events]
    assert names == ["e1", "e2"]


def test_file_sink_creates_dir(tmp_path):
    deep = tmp_path / "a" / "b" / "c"
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(deep))
    init_logging(cfg)
    new_session()
    get_logger("test").info("hello")
    assert deep.exists()
    assert _today_file(deep).exists()


def test_file_sink_expanduser(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    cfg = LoggingConfig(enabled=True, output="file", file_dir="~/logs")
    init_logging(cfg)
    new_session()
    get_logger("test").info("hello")
    assert (tmp_path / "logs").exists()


def test_console_sink_captures_event(capfd, tmp_path):
    cfg = LoggingConfig(enabled=True, output="console", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("test").info("tool.call", tool_name="read_file")
    out = capfd.readouterr()
    combined = out.out + out.err
    assert "tool.call" in combined
    assert "read_file" in combined


def test_both_output_writes_to_file_and_console(capfd, tmp_path):
    cfg = LoggingConfig(enabled=True, output="both", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    get_logger("test").info("agent.run", phase="end")

    # Console branch: event name + field rendered on stderr specifically
    out = capfd.readouterr()
    assert "agent.run" in out.err
    assert "phase" in out.err

    # File branch: valid JSONL with expected fields
    assert _today_file(tmp_path).exists()
    events = _read_jsonl(_today_file(tmp_path))
    assert events[0]["event"] == "agent.run"
    assert events[0]["phase"] == "end"


def test_file_sink_rotates_on_day_boundary(tmp_path, monkeypatch):
    """When the UTC date changes mid-session, subsequent events go to a new file."""
    from norn.observability import sinks

    fake_now = [datetime(2026, 4, 21, 23, 59, 59, tzinfo=UTC)]

    class _FakeDatetime:
        @staticmethod
        def now(tz=None):  # noqa: ARG004
            return fake_now[0]

    monkeypatch.setattr(sinks, "datetime", _FakeDatetime)

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    log = get_logger("test")

    log.info("before.midnight", marker="day1")

    # Advance past UTC midnight
    fake_now[0] = datetime(2026, 4, 22, 0, 0, 1, tzinfo=UTC)
    log.info("after.midnight", marker="day2")

    file_day1 = tmp_path / "2026-04-21.jsonl"
    file_day2 = tmp_path / "2026-04-22.jsonl"
    assert file_day1.exists(), "day1 file missing"
    assert file_day2.exists(), "day2 file missing"

    day1_events = _read_jsonl(file_day1)
    day2_events = _read_jsonl(file_day2)
    assert [e["event"] for e in day1_events] == ["before.midnight"]
    assert [e["event"] for e in day2_events] == ["after.midnight"]


def test_file_sink_json_valid(tmp_path):
    """Each line in the JSONL file must parse as valid JSON."""
    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    new_session()
    log = get_logger("test")
    log.info("llm.complete", model="gpt-4", prompt_tokens=10, completion_tokens=5)

    text = _today_file(tmp_path).read_text()
    for line in text.splitlines():
        if line.strip():
            json.loads(line)  # raises if invalid
