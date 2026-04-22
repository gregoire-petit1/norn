"""JSONL persistence for benchmark results."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.runner.models import RunMeta, RunReport, TaskResult


def generate_filename(meta: RunMeta) -> str:
    """Generate a JSONL filename from run metadata."""
    ts = meta.timestamp.strftime("%Y%m%d_%H%M%S")
    run_id = meta.run_id or "unknown"
    return f"{ts}_{run_id}.jsonl"


def save_report(report: RunReport, *, results_dir: Path) -> Path:
    """Save a RunReport as JSONL: meta header + task_result lines.

    Creates results_dir if it doesn't exist.
    Returns the path to the written file.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = generate_filename(report.meta)
    filepath = results_dir / filename

    with open(filepath, "w") as f:
        # Meta header line
        meta_line = {"type": "meta", **json.loads(report.meta.model_dump_json())}
        f.write(json.dumps(meta_line) + "\n")

        # Task result lines
        for result in report.results:
            result_line = {"type": "result", **json.loads(result.model_dump_json())}
            f.write(json.dumps(result_line) + "\n")

    return filepath


def load_report(filepath: Path) -> RunReport:
    """Load a RunReport from a JSONL file."""
    meta = None
    results: list[TaskResult] = []

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            line_type = data.pop("type", None)
            if line_type == "meta":
                meta = RunMeta.model_validate(data)
            elif line_type == "result":
                results.append(TaskResult.model_validate(data))

    return RunReport(
        meta=meta or RunMeta(),
        results=results,
    )


def list_reports(*, results_dir: Path) -> list[Path]:
    """List all JSONL report files in results_dir, sorted by name."""
    if not results_dir.is_dir():
        return []
    return sorted(results_dir.glob("*.jsonl"))
