"""Tests for benchmark task loader."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.runner.loader import load_tasks


def _write_task(base: Path, category: str, task_id: str, *, deprecated: bool = False) -> Path:
    """Create a minimal task.yaml in the correct directory structure."""
    task_dir = base / category / task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    # Create seed dir
    seed_dir = task_dir / "seed"
    seed_dir.mkdir(exist_ok=True)
    (seed_dir / "placeholder.py").write_text("# seed\n")

    data = {
        "id": f"{category}-{task_id}",
        "category": category,
        "description": f"Test task {task_id}",
        "prompt": f"Do {task_id}",
        "seed_dir": "seed/",
        "eval_command": "echo ok",
    }
    if deprecated:
        data["deprecated"] = True

    task_yaml = task_dir / "task.yaml"
    task_yaml.write_text(yaml.dump(data))
    return task_dir


def test_load_tasks_finds_all(tmp_path):
    """Loads all task.yaml files from nested directory structure."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "code-gen", "002-fizz")
    _write_task(tmp_path, "bug-fix", "001-off-by-one")

    tasks = load_tasks(tasks_dir=tmp_path)
    assert len(tasks) == 3
    ids = {t.id for t in tasks}
    assert "code-gen-001-fib" in ids
    assert "bug-fix-001-off-by-one" in ids


def test_load_tasks_skips_deprecated(tmp_path):
    """Deprecated tasks are skipped."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "code-gen", "002-old", deprecated=True)

    tasks = load_tasks(tasks_dir=tmp_path)
    assert len(tasks) == 1
    assert tasks[0].id == "code-gen-001-fib"


def test_load_tasks_filter_by_category(tmp_path):
    """Filter tasks by category."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "bug-fix", "001-off")

    tasks = load_tasks(tasks_dir=tmp_path, category="code-gen")
    assert len(tasks) == 1
    assert tasks[0].category == "code-gen"


def test_load_tasks_filter_by_task_id(tmp_path):
    """Filter tasks by specific task ID."""
    _write_task(tmp_path, "code-gen", "001-fib")
    _write_task(tmp_path, "code-gen", "002-fizz")

    tasks = load_tasks(tasks_dir=tmp_path, task_id="code-gen-001-fib")
    assert len(tasks) == 1
    assert tasks[0].id == "code-gen-001-fib"


def test_load_tasks_empty_dir(tmp_path):
    """Empty tasks dir returns empty list."""
    assert load_tasks(tasks_dir=tmp_path) == []


def test_load_tasks_nonexistent_dir(tmp_path):
    """Non-existent tasks dir returns empty list."""
    assert load_tasks(tasks_dir=tmp_path / "nonexistent") == []


def test_load_tasks_sets_source_path(tmp_path):
    """Loaded tasks have _source_path set."""
    _write_task(tmp_path, "code-gen", "001-fib")

    tasks = load_tasks(tasks_dir=tmp_path)
    assert len(tasks) == 1
    assert tasks[0]._source_path is not None
    assert tasks[0]._source_path.name == "task.yaml"
