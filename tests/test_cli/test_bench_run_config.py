"""Tests for the de-hardcoded bench run + judge wiring + guard (SOTA v2, workstream B)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
import yaml
from typer.testing import CliRunner

from benchmarks.runner.models import ExecutionTrace, RunMeta, RunReport, TaskResult
from norn.cli.bench import bench_app

if TYPE_CHECKING:
    from pathlib import Path

runner = CliRunner()


def _write_task(base: Path, category: str, task_id: str, *, difficulty: str | None = None) -> None:
    task_dir = base / category / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    seed = task_dir / "seed"
    seed.mkdir(exist_ok=True)
    (seed / "main.py").write_text("# seed\n")
    data = {
        "id": f"{category}-{task_id}",
        "category": category,
        "description": f"Test {task_id}",
        "prompt": f"Do {task_id}",
        "seed_dir": "seed/",
        "eval_command": "echo ok",
    }
    if difficulty:
        data["metadata"] = {"difficulty": difficulty}
    (task_dir / "task.yaml").write_text(yaml.dump(data))


@pytest.fixture
def fast_bench(monkeypatch):
    """No inter-task sleep + a stubbed executor capturing config_overrides."""
    monkeypatch.setenv(
        "NORN_CONFIG_OVERRIDES", json.dumps({"bench": {"inter_task_delay_seconds": 0}})
    )
    captured: dict = {}

    async def fake_execute_task(task, *, sandbox_dir, config_overrides, timeout_seconds=None):
        captured["config_overrides"] = config_overrides
        return ExecutionTrace(exit_code=0, stdout="agent done", duration_ms=10)

    import benchmarks.runner.executor as executor_mod

    monkeypatch.setattr(executor_mod, "execute_task", fake_execute_task)
    return captured


def _load_saved_report(results_dir: Path) -> list[dict]:
    files = sorted(results_dir.glob("*.jsonl"))
    assert files, f"no report saved in {results_dir}"
    lines = [json.loads(line) for line in files[-1].read_text().splitlines()]
    return [line for line in lines if line.get("task_id")]


def test_bench_run_tiers_come_from_config(tmp_path, fast_bench):
    _write_task(tmp_path / "tasks", "code-gen", "001")
    result = runner.invoke(
        bench_app,
        ["run", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    assert result.exit_code == 0, result.output
    tiers = fast_bench["config_overrides"]["router"]["tiers"]
    # Defaults preserved from BenchConfig (previously hardcoded values)
    assert tiers["fast"] == {
        "provider": "github_copilot",
        "model": "claude-sonnet-4.5",
        "api_base": None,
    }
    assert fast_bench["config_overrides"]["agent"]["max_tool_rounds"] == 20


def test_bench_run_model_option_drives_tiers(tmp_path, fast_bench):
    _write_task(tmp_path / "tasks", "code-gen", "001")
    result = runner.invoke(
        bench_app,
        [
            "run",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--results-dir",
            str(tmp_path / "res"),
            "--model",
            "groq/llama-3.3-70b-versatile",
        ],
    )
    assert result.exit_code == 0, result.output
    tiers = fast_bench["config_overrides"]["router"]["tiers"]
    for tier in tiers.values():
        assert tier["provider"] == "groq"
        assert tier["model"] == "llama-3.3-70b-versatile"
    assert "groq/llama-3.3-70b-versatile" in result.output


def test_bench_run_n_runs(tmp_path, fast_bench):
    _write_task(tmp_path / "tasks", "code-gen", "001")
    result = runner.invoke(
        bench_app,
        [
            "run",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--results-dir",
            str(tmp_path / "res"),
            "--n-runs",
            "2",
        ],
    )
    assert result.exit_code == 0, result.output
    results = _load_saved_report(tmp_path / "res")
    assert len(results) == 2


def test_bench_run_judge_opt_in(tmp_path, fast_bench, monkeypatch):
    _write_task(tmp_path / "tasks", "code-gen", "001")
    calls: list = []

    async def fake_judge_task(task, agent_output, *, llm, **kwargs):
        calls.append(agent_output)
        from benchmarks.runner.models import JudgeScore

        return JudgeScore(correctness=9, quality=8, completeness=9, reasoning="solid")

    import benchmarks.runner.judge as judge_mod

    monkeypatch.setattr(judge_mod, "judge_task", fake_judge_task)

    # Without --judge: never called
    result = runner.invoke(
        bench_app,
        ["run", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res1")],
    )
    assert result.exit_code == 0, result.output
    assert calls == []

    # With --judge: called, score persisted, summary printed
    result = runner.invoke(
        bench_app,
        [
            "run",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--results-dir",
            str(tmp_path / "res2"),
            "--judge",
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    results = _load_saved_report(tmp_path / "res2")
    assert results[0]["judge_score"]["correctness"] == 9
    assert "Judge avg" in result.output


def test_bench_run_warns_on_parallel_tasks(tmp_path, fast_bench, monkeypatch):
    monkeypatch.setenv(
        "NORN_CONFIG_OVERRIDES",
        json.dumps({"bench": {"inter_task_delay_seconds": 0, "parallel_tasks": 4}}),
    )
    _write_task(tmp_path / "tasks", "code-gen", "001")
    result = runner.invoke(
        bench_app,
        ["run", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    assert result.exit_code == 0, result.output
    assert "parallel_tasks" in result.output


# --------------------------------------------------------------------------- #
# norn bench guard (W3.3)
# --------------------------------------------------------------------------- #


def _synthetic_report(passing: dict[str, bool]) -> RunReport:
    return RunReport(
        meta=RunMeta(run_id="guard-test", model="stub"),
        results=[TaskResult(task_id=tid, success=ok) for tid, ok in passing.items()],
    )


@pytest.fixture
def guard_env(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "NORN_CONFIG_OVERRIDES", json.dumps({"bench": {"inter_task_delay_seconds": 0}})
    )
    _write_task(tmp_path / "tasks", "code-gen", "001", difficulty="easy")
    _write_task(tmp_path / "tasks", "code-gen", "002", difficulty="easy")
    _write_task(tmp_path / "tasks", "code-gen", "003-hard", difficulty="hard")
    return tmp_path


def _patch_guard_run(monkeypatch, report: RunReport) -> dict:
    import norn.cli.bench as bench_mod

    seen: dict = {}

    def fake_run_tasks(tasks, **kwargs):
        seen["task_ids"] = [t.id for t in tasks]
        return report

    monkeypatch.setattr(bench_mod, "_run_tasks", fake_run_tasks)
    return seen


def test_guard_records_baseline_when_none(guard_env, monkeypatch):
    tmp_path = guard_env
    seen = _patch_guard_run(
        monkeypatch, _synthetic_report({"code-gen-001": True, "code-gen-002": True})
    )
    result = runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    assert result.exit_code == 0, result.output
    assert "baseline recorded" in result.output.lower()
    # Only easy tasks selected
    assert seen["task_ids"] == ["code-gen-001", "code-gen-002"]
    assert list((tmp_path / "res" / "guard").glob("*.jsonl"))


def test_guard_passes_without_regression(guard_env, monkeypatch):
    tmp_path = guard_env
    _patch_guard_run(monkeypatch, _synthetic_report({"code-gen-001": True}))
    runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    result = runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    assert result.exit_code == 0, result.output
    assert "no regressions" in result.output.lower()


def test_guard_exits_1_on_regression(guard_env, monkeypatch):
    tmp_path = guard_env
    _patch_guard_run(monkeypatch, _synthetic_report({"code-gen-001": True}))
    runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    _patch_guard_run(monkeypatch, _synthetic_report({"code-gen-001": False}))
    result = runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    assert result.exit_code == 1
    assert "code-gen-001" in result.output


def test_guard_revert_lessons_on_regression(guard_env, monkeypatch):
    tmp_path = guard_env
    memory_dir = tmp_path / "memory"
    monkeypatch.setenv(
        "NORN_CONFIG_OVERRIDES",
        json.dumps(
            {
                "bench": {"inter_task_delay_seconds": 0},
                "memory": {"enabled": True, "memory_dir": str(memory_dir)},
            }
        ),
    )
    # Simulate /reflect: backup, then persist a (bad) lesson
    from norn.memory.models import MemoryConfig
    from norn.memory.store import MemoryStore

    store = MemoryStore(MemoryConfig(memory_dir=memory_dir))
    store.ensure_dirs()
    store.append_lesson("## Good lesson\n\nkeep me")
    store.backup_lessons()
    store.append_lesson("## Bad lesson\n\nregression source")

    _patch_guard_run(monkeypatch, _synthetic_report({"code-gen-001": True}))
    runner.invoke(
        bench_app,
        ["guard", "--tasks-dir", str(tmp_path / "tasks"), "--results-dir", str(tmp_path / "res")],
    )
    _patch_guard_run(monkeypatch, _synthetic_report({"code-gen-001": False}))
    result = runner.invoke(
        bench_app,
        [
            "guard",
            "--tasks-dir",
            str(tmp_path / "tasks"),
            "--results-dir",
            str(tmp_path / "res"),
            "--revert-lessons",
        ],
    )
    assert result.exit_code == 1
    assert "restored" in result.output.lower()
    lessons = store.read_lessons()
    assert "keep me" in lessons
    assert "regression source" not in lessons
