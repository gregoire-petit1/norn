"""Tests for JSONL benchmark result persistence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from benchmarks.runner.models import RunMeta, RunReport, TaskResult
from benchmarks.runner.persistence import (
    save_report,
    load_report,
    list_reports,
    generate_filename,
)


def _make_report(n_results: int = 2) -> RunReport:
    meta = RunMeta(
        run_id="test-run-001",
        model="test-model",
        agent_version="0.1.0",
        timestamp=datetime(2026, 4, 22, 12, 0, 0, tzinfo=timezone.utc),
    )
    results = [
        TaskResult(task_id=f"task-{i:03d}", success=i % 2 == 0, latency_ms=1000 + i * 100)
        for i in range(n_results)
    ]
    return RunReport(meta=meta, results=results)


def test_generate_filename():
    """Filename includes timestamp and run_id."""
    meta = RunMeta(
        run_id="abc123",
        timestamp=datetime(2026, 4, 22, 14, 30, 0, tzinfo=timezone.utc),
    )
    name = generate_filename(meta)
    assert "2026" in name
    assert "abc123" in name
    assert name.endswith(".jsonl")


def test_save_and_load_roundtrip(tmp_path):
    """Save report then load it back — data must be identical."""
    report = _make_report(3)
    filepath = save_report(report, results_dir=tmp_path)

    assert filepath.exists()
    assert filepath.suffix == ".jsonl"

    loaded = load_report(filepath)
    assert loaded.meta.run_id == report.meta.run_id
    assert loaded.meta.model == report.meta.model
    assert len(loaded.results) == 3
    assert loaded.results[0].task_id == "task-000"
    assert loaded.results[0].success is True
    assert loaded.results[1].success is False


def test_save_creates_results_dir(tmp_path):
    """Save auto-creates results_dir if missing."""
    results_dir = tmp_path / "nested" / "results"
    report = _make_report(1)
    filepath = save_report(report, results_dir=results_dir)
    assert filepath.exists()
    assert results_dir.is_dir()


def test_jsonl_format(tmp_path):
    """Each line in the JSONL file is valid JSON."""
    report = _make_report(2)
    filepath = save_report(report, results_dir=tmp_path)

    lines = filepath.read_text().strip().splitlines()
    # First line: meta header, then task results
    assert len(lines) >= 3  # meta + 2 results
    for line in lines:
        data = json.loads(line)
        assert isinstance(data, dict)
    # First line should have "type": "meta"
    first = json.loads(lines[0])
    assert first.get("type") == "meta"


def test_list_reports(tmp_path):
    """list_reports finds all JSONL files in results_dir."""
    r1 = _make_report(1)
    r1.meta.run_id = "run-001"
    r2 = _make_report(1)
    r2.meta.run_id = "run-002"

    save_report(r1, results_dir=tmp_path)
    save_report(r2, results_dir=tmp_path)

    reports = list_reports(results_dir=tmp_path)
    assert len(reports) == 2
    assert all(p.suffix == ".jsonl" for p in reports)


def test_list_reports_empty_dir(tmp_path):
    """list_reports returns empty list for empty dir."""
    assert list_reports(results_dir=tmp_path) == []


def test_list_reports_missing_dir(tmp_path):
    """list_reports returns empty list for non-existent dir."""
    assert list_reports(results_dir=tmp_path / "nonexistent") == []
