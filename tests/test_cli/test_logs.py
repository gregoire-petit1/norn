"""Tests for ``norn logs tail`` (Phase 9 F1).

Pattern: seed a JSONL fixture under ``tmp_path``, monkeypatch
``norn.cli.logs._today_log_path`` to point at it, drive the command via
``CliRunner`` with ``mix_stderr=False`` so we can assert on stdout vs stderr
independently (default behavior, malformed warnings, alias resolution errors).
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from typer.testing import CliRunner

from norn.cli.main import app

if TYPE_CHECKING:
    from pathlib import Path


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed_jsonl(path: Path, lines: list[dict]) -> None:
    """Write a JSONL fixture, creating parent dirs as needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(d) for d in lines) + "\n", encoding="utf-8")


def _make_runner() -> CliRunner:
    """CliRunner with default stderr separation (Click >= 8.2 / Typer >= 0.13)."""
    return CliRunner()


_SAMPLE_SID_A = "11111111-2222-3333-4444-555555555555"
_SAMPLE_SID_B = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"


def _sample_events() -> list[dict]:
    return [
        {
            "event": "agent.run",
            "level": "info",
            "timestamp": "2026-04-22T10:00:00Z",
            "session_id": _SAMPLE_SID_A,
            "phase": "start",
        },
        {
            "event": "tool.call",
            "level": "info",
            "timestamp": "2026-04-22T10:00:01Z",
            "session_id": _SAMPLE_SID_A,
            "tool": "bash",
        },
        {
            "event": "llm.complete",
            "level": "info",
            "timestamp": "2026-04-22T10:00:02Z",
            "session_id": _SAMPLE_SID_B,
            "tokens": 42,
        },
    ]


# ---------------------------------------------------------------------------
# 1. Missing log file
# ---------------------------------------------------------------------------


def test_tail_no_logs_today_prints_info(tmp_path, monkeypatch):
    """Missing file → exit 0 with an informational stderr message."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"  # never created
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(app, ["logs", "tail"])

    assert result.exit_code == 0
    assert result.stdout == ""
    assert "No logs for today" in result.stderr
    assert str(log_path) in result.stderr


# ---------------------------------------------------------------------------
# 2. Default rendering
# ---------------------------------------------------------------------------


def test_tail_default_emits_all_events(tmp_path, monkeypatch):
    """3 events seeded → 3 formatted lines on stdout, exit 0."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(app, ["logs", "tail"])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 3
    # Each line must contain the event name and an HH:MM:SS prefix.
    assert "agent.run" in result.stdout
    assert "tool.call" in result.stdout
    assert "llm.complete" in result.stdout
    assert "10:00:00" in result.stdout


# ---------------------------------------------------------------------------
# 3. Single --event filter
# ---------------------------------------------------------------------------


def test_tail_filter_event_single(tmp_path, monkeypatch):
    """``--event tool.call`` → only that event."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(app, ["logs", "tail", "--event", "tool.call"])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "tool.call" in result.stdout
    assert "agent.run" not in result.stdout
    assert "llm.complete" not in result.stdout


# ---------------------------------------------------------------------------
# 4. Multi --event OR semantics
# ---------------------------------------------------------------------------


def test_tail_filter_event_multiple_or(tmp_path, monkeypatch):
    """``--event tool.call --event llm.complete`` → OR (2 lines)."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(
        app,
        ["logs", "tail", "-e", "tool.call", "-e", "llm.complete"],
    )

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "tool.call" in result.stdout
    assert "llm.complete" in result.stdout
    assert "agent.run" not in result.stdout


# ---------------------------------------------------------------------------
# 5. Exact UUID --session match
# ---------------------------------------------------------------------------


def test_tail_filter_session_uuid_exact(tmp_path, monkeypatch):
    """``--session <full-uuid>`` returns only events with that session_id."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(app, ["logs", "tail", "--session", _SAMPLE_SID_B])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "llm.complete" in result.stdout
    assert "tool.call" not in result.stdout


# ---------------------------------------------------------------------------
# 6. --session last → resolves via state file
# ---------------------------------------------------------------------------


def test_tail_filter_session_last_resolves_state_file(tmp_path, monkeypatch):
    """Seed the state file → ``--session last`` resolves to that uuid."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "last_session").write_text(_SAMPLE_SID_A, encoding="utf-8")
    monkeypatch.setattr(
        "norn.cli.logs._LAST_SESSION_FILE",
        state_dir / "last_session",
    )

    result = _make_runner().invoke(app, ["logs", "tail", "--session", "last"])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 2  # both SID_A events
    assert "agent.run" in result.stdout
    assert "tool.call" in result.stdout
    assert "llm.complete" not in result.stdout


# ---------------------------------------------------------------------------
# 7. --session last with no state file → BadParameter (exit 2)
# ---------------------------------------------------------------------------


def test_tail_filter_session_last_missing_state_errors(tmp_path, monkeypatch):
    """No state file → typer.BadParameter → exit 2."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)
    monkeypatch.setattr(
        "norn.cli.logs._LAST_SESSION_FILE",
        tmp_path / "state" / "last_session",  # parent dir missing
    )

    result = _make_runner().invoke(app, ["logs", "tail", "--session", "last"])

    assert result.exit_code == 2
    assert "No previous session found" in result.stderr


# ---------------------------------------------------------------------------
# 8. --session current alias
# ---------------------------------------------------------------------------


def test_tail_filter_session_current_alias(tmp_path, monkeypatch):
    """``--session current`` is an alias for ``--session last``."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "last_session").write_text(_SAMPLE_SID_B, encoding="utf-8")
    monkeypatch.setattr(
        "norn.cli.logs._LAST_SESSION_FILE",
        state_dir / "last_session",
    )

    result = _make_runner().invoke(app, ["logs", "tail", "--session", "current"])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "llm.complete" in result.stdout


# ---------------------------------------------------------------------------
# 9. --json passthrough
# ---------------------------------------------------------------------------


def test_tail_json_passthrough(tmp_path, monkeypatch):
    """``--json`` emits raw JSONL post-filter (parseable by jq)."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(
        app,
        ["logs", "tail", "--event", "tool.call", "--json"],
    )

    assert result.exit_code == 0, result.stderr
    raw_lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(raw_lines) == 1
    parsed = json.loads(raw_lines[0])
    # Full UUID preserved (not truncated like rich mode).
    assert parsed["session_id"] == _SAMPLE_SID_A
    assert parsed["event"] == "tool.call"
    assert parsed["tool"] == "bash"


# ---------------------------------------------------------------------------
# 10. Malformed lines skipped + aggregated stderr count
# ---------------------------------------------------------------------------


def test_tail_skips_malformed_lines(tmp_path, monkeypatch):
    """Invalid JSON lines are dropped silently, count emitted to stderr."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    valid = _sample_events()[0]
    log_path.write_text(
        "\n".join(
            [
                json.dumps(valid),
                "{not valid json",
                "garbage line without braces",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    result = _make_runner().invoke(app, ["logs", "tail"])

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "agent.run" in result.stdout
    assert "Skipped 2 malformed line(s)." in result.stderr


# ---------------------------------------------------------------------------
# 11. Combined --event + --session filters use AND
# ---------------------------------------------------------------------------


def test_tail_combined_filters_and_logic(tmp_path, monkeypatch):
    """``--event X --session Y`` requires both: AND between filter axes."""
    log_path = tmp_path / "logs" / "2026-04-22.jsonl"
    _seed_jsonl(log_path, _sample_events())
    monkeypatch.setattr("norn.cli.logs._today_log_path", lambda: log_path)

    # SID_A has agent.run and tool.call; combining with --event tool.call
    # must yield exactly 1 line (not 2 = SID_A events, not 1 = tool.call alone
    # if logic were OR'd incorrectly).
    result = _make_runner().invoke(
        app,
        [
            "logs",
            "tail",
            "--event",
            "tool.call",
            "--session",
            _SAMPLE_SID_A,
        ],
    )

    assert result.exit_code == 0, result.stderr
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert len(lines) == 1
    assert "tool.call" in result.stdout
    assert "agent.run" not in result.stdout
    assert "llm.complete" not in result.stdout
