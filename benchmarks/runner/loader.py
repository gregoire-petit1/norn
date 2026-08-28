"""Task loader: walks task directories and loads TaskDefs."""

from __future__ import annotations

from pathlib import Path

import yaml

from benchmarks.runner.models import TaskDef


def load_tasks(
    *,
    tasks_dir: Path,
    category: str | None = None,
    task_id: str | None = None,
    difficulty: str | None = None,
) -> list[TaskDef]:
    """Load all TaskDefs from tasks_dir/**/task.yaml.

    Args:
        tasks_dir: Root directory containing task subdirectories.
        category: Optional filter by category.
        task_id: Optional filter by specific task ID.
        difficulty: Optional filter by metadata.difficulty (e.g. "easy").

    Returns:
        List of TaskDef, sorted by id. Deprecated tasks are excluded.
    """
    if not tasks_dir.is_dir():
        return []

    tasks: list[TaskDef] = []
    for task_yaml in sorted(tasks_dir.rglob("task.yaml")):
        # Peek at YAML to check deprecated flag before full parse
        try:
            with open(task_yaml) as f:
                raw = yaml.safe_load(f) or {}
        except Exception:
            continue

        if raw.get("deprecated", False):
            continue

        try:
            td = TaskDef.from_yaml(task_yaml)
        except Exception:
            continue

        if category and td.category != category:
            continue
        if task_id and td.id != task_id:
            continue
        if difficulty and td.metadata.difficulty != difficulty:
            continue

        tasks.append(td)

    return sorted(tasks, key=lambda t: t.id)
