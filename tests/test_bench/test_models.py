import pytest
from benchmarks.runner.models import TaskDef, TaskResult, ExecutionTrace, JudgeScore, RunMeta


def test_taskdef_from_yaml(tmp_path):
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("""
id: "code-gen-001-fibonacci"
category: "code-gen"
description: "Implement fibonacci"
prompt: "Write fib(n)"
seed_dir: "seed/"
eval_command: "uv run pytest tests/ -v"
timeout_seconds: 120
""")
    td = TaskDef.from_yaml(task_yaml)
    assert td.id == "code-gen-001-fibonacci"
    assert td.category == "code-gen"
    assert td.timeout_seconds == 120
    assert td.n_runs == 1  # default


def test_taskdef_missing_required_field(tmp_path):
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("id: test\ncategory: code-gen\n")
    with pytest.raises(Exception):
        TaskDef.from_yaml(task_yaml)


def test_taskdef_default_n_runs(tmp_path):
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("""
id: test
category: code-gen
description: d
prompt: p
seed_dir: seed/
eval_command: echo ok
""")
    td = TaskDef.from_yaml(task_yaml)
    assert td.n_runs == 1


def test_taskdef_absolute_seed_dir(tmp_path):
    task_yaml = tmp_path / "task.yaml"
    task_yaml.write_text("""
id: test
category: code-gen
description: d
prompt: p
seed_dir: seed/
eval_command: echo ok
""")
    td = TaskDef.from_yaml(task_yaml)
    assert td.absolute_seed_dir == tmp_path / "seed"


def test_execution_trace_timeout():
    trace = ExecutionTrace.timeout(300)
    assert trace.timed_out is True
    assert trace.duration_ms == 300_000


def test_judge_score_passes():
    score = JudgeScore(correctness=8, quality=7, completeness=7, reasoning="ok")
    assert score.passes(threshold=7.0) is True


def test_judge_score_fails():
    score = JudgeScore(correctness=5, quality=7, completeness=7, reasoning="bad")
    assert score.passes(threshold=7.0) is False


def test_task_result_timeout():
    from benchmarks.runner.models import TaskDef, ExecutionTrace

    task = TaskDef(
        id="t", category="c", description="d", prompt="p", seed_dir="s", eval_command="e"
    )
    trace = ExecutionTrace.timeout(120)
    result = TaskResult.timeout(task, trace)
    assert result.timed_out is True
    assert result.task_id == "t"
    assert result.latency_ms == 120_000
