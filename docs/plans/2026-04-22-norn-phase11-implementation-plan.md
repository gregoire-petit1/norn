# Phase 11: Internal Mini-Eval Benchmark — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a benchmark runner that executes curated tasks against Norn, collects metrics, and reports regressions — giving us a measurable baseline before any optimization.

**Architecture:** Tasks defined as YAML + seed repos. Runner creates sandboxed tmpdir per task, invokes `norn run`, evaluates via `eval_command`, optionally judges via LLM. Results stored as append-only JSONL. CLI sub-app `norn bench {run,report,list,diff,validate}`.

**Tech Stack:** Python, Pydantic, asyncio, structlog, Rich (tables/markdown), PyYAML, existing LiteLLM provider for judge.

---

## Prerequisites

- Phase 9 v2 complete (prompt caching metrics available) ✅
- UX cleanup landed (clean CLI output) ✅
- `NORN_CONFIG_OVERRIDES` env var support must be added (Task 1)

---

## Session 1: Foundation — Config, Models, Sandbox

### Task 1: Add NORN_CONFIG_OVERRIDES env var support

**Files:**
- Modify: `src/norn/core/config.py:116-132`
- Test: `tests/test_core/test_config.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_config.py — append

def test_config_overrides_from_env(monkeypatch, tmp_path):
    """NORN_CONFIG_OVERRIDES env var applies nested key overrides."""
    import json
    overrides = {"logging": {"file_dir": str(tmp_path), "output": "file"}, "permissions": {"mode": "strict"}}
    monkeypatch.setenv("NORN_CONFIG_OVERRIDES", json.dumps(overrides))
    cfg = NornConfig()
    cfg.apply_env_overrides()
    assert cfg.logging.file_dir == str(tmp_path)
    assert cfg.logging.output == "file"
    assert cfg.permissions.mode.value == "strict"


def test_config_overrides_invalid_json_ignored(monkeypatch):
    """Malformed NORN_CONFIG_OVERRIDES is silently ignored."""
    monkeypatch.setenv("NORN_CONFIG_OVERRIDES", "not-json{{{")
    cfg = NornConfig()
    cfg.apply_env_overrides()  # must not raise
    assert cfg.logging.output == "both"  # default unchanged
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_config.py::test_config_overrides_from_env -v`
Expected: FAIL

**Step 3: Implement**

In `src/norn/core/config.py`, add at the end of `apply_env_overrides()`:

```python
        # Generic JSON overrides (used by bench runner to inject per-task config)
        raw_overrides = os.environ.get("NORN_CONFIG_OVERRIDES")
        if raw_overrides:
            import json as _json
            try:
                overrides = _json.loads(raw_overrides)
            except (ValueError, TypeError):
                return  # malformed JSON silently ignored
            if isinstance(overrides, dict):
                merged = self.model_dump()
                for section, fields in overrides.items():
                    if section in merged and isinstance(fields, dict):
                        merged[section] = {**merged[section], **fields}
                    elif section in merged:
                        merged[section] = fields
                updated = type(self).model_validate(merged)
                for field_name in self.model_fields:
                    setattr(self, field_name, getattr(updated, field_name))
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/norn/core/config.py tests/test_core/test_config.py
git commit -m "feat(config): add NORN_CONFIG_OVERRIDES env var for nested overrides"
```

---

### Task 2: BenchConfig + FlagsConfig.bench

**Files:**
- Modify: `src/norn/core/config.py`
- Modify: `configs/default.yaml`
- Test: `tests/test_core/test_config.py`

**Step 1: Write the failing test**

```python
def test_bench_config_defaults():
    from norn.core.config import BenchConfig
    cfg = BenchConfig()
    assert cfg.tasks_dir == "benchmarks/tasks"
    assert cfg.results_dir == "benchmarks/results"
    assert cfg.default_n_runs == 1
    assert cfg.default_timeout_seconds == 300
    assert cfg.judge.enabled is True
    assert cfg.judge.pass_threshold == 7.0


def test_bench_flag_default_false():
    cfg = NornConfig()
    assert cfg.flags.bench is False
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_core/test_config.py::test_bench_config_defaults -v`
Expected: FAIL (BenchConfig doesn't exist)

**Step 3: Implement**

Add to `src/norn/core/config.py`:

```python
class JudgeConfig(BaseModel):
    enabled: bool = True
    model: str = "anthropic/claude-sonnet-4"
    pass_threshold: float = 7.0


class BenchConfig(BaseModel):
    tasks_dir: str = "benchmarks/tasks"
    results_dir: str = "benchmarks/results"
    reports_dir: str = "benchmarks/reports"
    default_n_runs: int = 1
    default_timeout_seconds: int = 300
    judge: JudgeConfig = JudgeConfig()
    parallel_tasks: int = 1
```

Add `bench: bool = False` to `FlagsConfig`.
Add `bench: BenchConfig = BenchConfig()` to `NornConfig`.
Add corresponding block to `configs/default.yaml`.

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_config.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add src/norn/core/config.py configs/default.yaml tests/test_core/test_config.py
git commit -m "feat(config): add BenchConfig and bench feature flag"
```

---

### Task 3: Benchmark models (TaskDef, TaskResult, ExecutionTrace, RunReport)

**Files:**
- Create: `benchmarks/runner/__init__.py`
- Create: `benchmarks/runner/models.py`
- Test: `tests/test_bench/test_models.py`
- Create: `tests/test_bench/__init__.py`

**Step 1: Write failing tests**

```python
# tests/test_bench/test_models.py
import pytest
from benchmarks.runner.models import TaskDef, TaskResult, ExecutionTrace, JudgeScore


def test_taskdef_from_yaml(tmp_path):
    """TaskDef loads from a valid YAML file."""
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
    with pytest.raises(Exception):  # Pydantic ValidationError
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


def test_execution_trace_timeout():
    trace = ExecutionTrace.timeout(300)
    assert trace.timed_out is True
    assert trace.duration_ms == 300_000


def test_judge_score_passes():
    score = JudgeScore(correctness=8, quality=7, completeness=7, reasoning="ok")
    assert score.passes(threshold=7.0) is True
    score2 = JudgeScore(correctness=5, quality=7, completeness=7, reasoning="bad")
    assert score2.passes(threshold=7.0) is False
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_bench/test_models.py -v`
Expected: FAIL (module not found)

**Step 3: Implement `benchmarks/runner/models.py`**

```python
"""Benchmark data models."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TaskMetadata(BaseModel):
    difficulty: str = "medium"
    expected_turns: int = 3
    uses_tools: list[str] = Field(default_factory=list)
    judge_required: bool = False


class TaskDef(BaseModel):
    id: str
    category: str
    description: str
    prompt: str
    seed_dir: str
    eval_command: str
    expected_files_changed: list[str] = Field(default_factory=list)
    forbidden_files_changed: list[str] = Field(default_factory=list)
    timeout_seconds: int = 300
    n_runs: int = 1
    metadata: TaskMetadata = TaskMetadata()
    deprecated: bool = False

    # Set at load time, not in YAML
    base_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> TaskDef:
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        td = cls(**data)
        td.base_path = path.parent
        return td

    @property
    def absolute_seed_dir(self) -> Path:
        if self.base_path is None:
            return Path(self.seed_dir)
        return self.base_path / self.seed_dir


class JudgeScore(BaseModel):
    correctness: float
    quality: float
    completeness: float
    reasoning: str

    def passes(self, threshold: float = 7.0) -> bool:
        avg = (self.correctness + self.quality + self.completeness) / 3
        return avg >= threshold


class ExecutionTrace(BaseModel):
    events: list[dict[str, Any]] = Field(default_factory=list)
    duration_ms: float = 0.0
    stdout: str = ""
    stderr: str = ""
    files_changed: list[str] = Field(default_factory=list)
    timed_out: bool = False

    @classmethod
    def timeout(cls, timeout_seconds: int) -> ExecutionTrace:
        return cls(timed_out=True, duration_ms=timeout_seconds * 1000)


class TaskResult(BaseModel):
    task_id: str
    category: str
    run_idx: int = 0
    success: bool = False
    binary_pass: bool = False
    judge_score: JudgeScore | None = None
    total_turns: int = 0
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0
    latency_ms: float = 0.0
    tool_calls_distribution: dict[str, int] = Field(default_factory=dict)
    permission_denials: int = 0
    eval_stdout: str = ""
    eval_returncode: int = -1
    files_changed: list[str] = Field(default_factory=list)
    timed_out: bool = False

    @classmethod
    def timeout(cls, task: TaskDef, trace: ExecutionTrace) -> TaskResult:
        return cls(
            task_id=task.id,
            category=task.category,
            timed_out=True,
            latency_ms=trace.duration_ms,
        )


class RunMeta(BaseModel):
    timestamp: str
    git_sha: str
    norn_version: str = "0.1.0"
    model: str
    n_runs: int
    n_tasks: int


class RunReport(BaseModel):
    meta: RunMeta
    results: list[TaskResult] = Field(default_factory=list)
    overall_success_rate: float = 0.0
    by_category: dict[str, float] = Field(default_factory=dict)
    total_tokens: int = 0
    total_cache_read_tokens: int = 0
```

Also create `benchmarks/__init__.py` and `benchmarks/runner/__init__.py` as empty files.

**Step 4: Run tests**

Run: `uv run pytest tests/test_bench/test_models.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add benchmarks/ tests/test_bench/
git commit -m "feat(bench): add benchmark data models (TaskDef, TaskResult, ExecutionTrace)"
```

---

### Task 4: Sandbox creation and cleanup

**Files:**
- Create: `benchmarks/runner/sandbox.py`
- Test: `tests/test_bench/test_sandbox.py`

**Step 1: Write failing tests**

```python
# tests/test_bench/test_sandbox.py
from pathlib import Path
from benchmarks.runner.sandbox import create_sandbox, cleanup_sandbox
from benchmarks.runner.models import TaskDef


def _make_seed(tmp_path: Path) -> Path:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "src").mkdir()
    (seed / "src" / "solution.py").write_text("# placeholder\n")
    (seed / "tests").mkdir()
    (seed / "tests" / "test_it.py").write_text("def test_pass(): pass\n")
    return seed


def test_create_sandbox_copies_files(tmp_path):
    seed = _make_seed(tmp_path)
    task = TaskDef(
        id="t1", category="c", description="d", prompt="p",
        seed_dir=str(seed), eval_command="echo ok",
    )
    task.base_path = tmp_path  # won't use absolute_seed_dir since seed_dir is absolute
    sandbox = create_sandbox(task, seed_dir_override=seed)
    assert (sandbox / "src" / "solution.py").exists()
    assert (sandbox / ".git").is_dir()
    cleanup_sandbox(sandbox)
    assert not sandbox.exists()


def test_sandbox_has_git_commit(tmp_path):
    seed = _make_seed(tmp_path)
    task = TaskDef(
        id="t2", category="c", description="d", prompt="p",
        seed_dir=str(seed), eval_command="echo ok",
    )
    sandbox = create_sandbox(task, seed_dir_override=seed)
    import subprocess
    result = subprocess.run(
        ["git", "log", "--oneline"], cwd=sandbox, capture_output=True, text=True,
    )
    assert "seed" in result.stdout
    cleanup_sandbox(sandbox)


def test_cleanup_sandbox_missing_path_no_error(tmp_path):
    cleanup_sandbox(tmp_path / "nonexistent")  # must not raise
```

**Step 2: Run to verify fail**

Run: `uv run pytest tests/test_bench/test_sandbox.py -v`
Expected: FAIL

**Step 3: Implement `benchmarks/runner/sandbox.py`**

```python
"""Sandbox creation: isolated tmpdir with git for each benchmark task."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from benchmarks.runner.models import TaskDef


def create_sandbox(task: TaskDef, *, seed_dir_override: Path | None = None) -> Path:
    """Copy seed/ to a tmpdir, init git, return sandbox path."""
    seed = seed_dir_override or task.absolute_seed_dir
    tmpdir = Path(tempfile.mkdtemp(prefix=f"norn-bench-{task.id}-"))
    shutil.copytree(seed, tmpdir, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=tmpdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmpdir, check=True)
    subprocess.run(
        ["git", "-c", "user.email=bench@norn", "-c", "user.name=Bench",
         "commit", "-qm", "seed"],
        cwd=tmpdir, check=True,
    )
    return tmpdir


def cleanup_sandbox(path: Path) -> None:
    """Remove the sandbox directory. No-op if path doesn't exist."""
    shutil.rmtree(path, ignore_errors=True)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_bench/test_sandbox.py -v`
Expected: ALL PASS

**Step 5: Commit**

```bash
git add benchmarks/runner/sandbox.py tests/test_bench/test_sandbox.py
git commit -m "feat(bench): add sandbox creation and cleanup"
```

---

## Session 2: Executor, Evaluator, Metrics, Judge

### Task 5: Metrics extraction helpers

**Files:**
- Create: `benchmarks/runner/metrics.py`
- Test: `tests/test_bench/test_metrics.py`

**Step 1: Write failing tests**

```python
# tests/test_bench/test_metrics.py
from benchmarks.runner.metrics import (
    count_events, sum_field, count_by_tool, count_events_by,
)

EVENTS = [
    {"event": "llm.complete", "total_tokens": 100, "prompt_tokens": 80, "completion_tokens": 20},
    {"event": "llm.complete", "total_tokens": 200, "prompt_tokens": 150, "completion_tokens": 50},
    {"event": "tool.call", "tool_name": "bash", "success": True},
    {"event": "tool.call", "tool_name": "file_read", "success": True},
    {"event": "tool.call", "tool_name": "bash", "success": True},
    {"event": "tool.call", "tool_name": "bash", "error_type": "PermissionDenied", "success": False},
]


def test_count_events():
    assert count_events(EVENTS, "llm.complete") == 2
    assert count_events(EVENTS, "tool.call") == 4


def test_sum_field():
    assert sum_field(EVENTS, "total_tokens") == 300
    assert sum_field(EVENTS, "prompt_tokens") == 230


def test_count_by_tool():
    dist = count_by_tool(EVENTS)
    assert dist == {"bash": 3, "file_read": 1}


def test_count_events_by():
    n = count_events_by(
        EVENTS, "tool.call", lambda e: e.get("error_type") == "PermissionDenied",
    )
    assert n == 1
```

**Step 2:** Run to verify fail. **Step 3:** Implement. **Step 4:** Run to verify pass. **Step 5:** Commit.

```bash
git commit -m "feat(bench): add metrics extraction helpers"
```

---

### Task 6: Executor (invokes Norn against task in sandbox)

**Files:**
- Create: `benchmarks/runner/executor.py`
- Test: `tests/test_bench/test_executor.py`

The executor uses `asyncio.create_subprocess_exec` to run `norn run <prompt>` in the sandbox. Tests mock the subprocess to avoid real LLM calls.

**Key test scenarios:**
1. Successful execution returns ExecutionTrace with events parsed from JSONL
2. Timeout kills process and returns `ExecutionTrace.timeout()`
3. Config overrides propagated via `NORN_CONFIG_OVERRIDES` env var
4. Non-zero exit code captured in stderr

**Step 5: Commit**

```bash
git commit -m "feat(bench): add task executor with subprocess + timeout"
```

---

### Task 7: Evaluator (runs eval_command, extracts metrics)

**Files:**
- Create: `benchmarks/runner/evaluator.py`
- Test: `tests/test_bench/test_evaluator.py`

Runs `task.eval_command` in the sandbox, captures returncode, extracts metrics from trace events using the helpers from Task 5.

**Key test scenarios:**
1. Passing eval_command → `binary_pass=True`, success=True
2. Failing eval_command → `binary_pass=False`, success=False
3. Timed-out trace → TaskResult.timeout()
4. Metrics correctly extracted from synthetic events
5. `files_changed` populated from trace

**Step 5: Commit**

```bash
git commit -m "feat(bench): add task evaluator with metric extraction"
```

---

### Task 8: LLM-as-Judge

**Files:**
- Create: `benchmarks/runner/judge.py`
- Test: `tests/test_bench/test_judge.py`

**Key test scenarios:**
1. Valid JSON response parsed into JudgeScore
2. Malformed JSON response returns None (graceful failure)
3. Score threshold logic (passes/fails)
4. Judge not called when `task.metadata.judge_required` is False

Tests mock the LLM provider — no real API calls.

**Step 5: Commit**

```bash
git commit -m "feat(bench): add LLM-as-judge evaluator"
```

---

## Session 3: CLI Sub-App, JSONL Persistence, Reporter

### Task 9: JSONL result persistence

**Files:**
- Create: `benchmarks/runner/persistence.py`
- Test: `tests/test_bench/test_persistence.py`

Write/read JSONL result files: `{timestamp}_{git_sha}.jsonl`.
Format: `run_meta` header + `task_result` lines + `run_summary` footer.

**Key test scenarios:**
1. Write results → read back identical data
2. File naming includes timestamp and git SHA
3. Append-only: existing files not modified
4. Handle missing results_dir (auto-create)

**Step 5: Commit**

```bash
git commit -m "feat(bench): add JSONL result persistence"
```

---

### Task 10: Reporter (Rich tables + markdown)

**Files:**
- Create: `benchmarks/runner/reporter.py`
- Test: `tests/test_bench/test_reporter.py`

Generate Rich table for terminal display + markdown for file output.
Includes: per-category success rate, aggregate metrics, regression detection.

**Key test scenarios:**
1. Report from results renders without error
2. Baseline comparison highlights regressions (success→fail)
3. Token bloat detection (>20% increase flagged)
4. Markdown output is valid markdown

**Step 5: Commit**

```bash
git commit -m "feat(bench): add Rich table + markdown reporter"
```

---

### Task 11: Task loader (walks tasks directory)

**Files:**
- Create: `benchmarks/runner/loader.py`
- Test: `tests/test_bench/test_loader.py`

Walks `benchmarks/tasks/**/task.yaml`, loads all TaskDefs, validates, filters deprecated.

**Key test scenarios:**
1. Loads multiple tasks from nested directory structure
2. Skips deprecated tasks
3. Filters by category
4. Filters by single task ID

**Step 5: Commit**

```bash
git commit -m "feat(bench): add task loader with category/id filtering"
```

---

### Task 12: CLI sub-app `norn bench`

**Files:**
- Create: `src/norn/cli/bench.py`
- Modify: `src/norn/cli/main.py` (register sub-app)
- Test: `tests/test_cli/test_bench_cli.py`

Typer sub-app with commands: `run`, `report`, `list`, `diff`, `validate`.

**Commands:**
- `norn bench run` — load tasks, create sandboxes, execute, evaluate, persist results
- `norn bench report` — load latest results, render Rich table
- `norn bench list` — list available tasks with metadata
- `norn bench diff <run_a> <run_b>` — compare two result files
- `norn bench validate` — lint all task.yaml files

**Key test scenarios:**
1. `bench list` renders task table
2. `bench validate` catches invalid task YAML
3. Smoke tests for each command (mocked execution)

**Step 5: Commit**

```bash
git commit -m "feat(cli): add norn bench sub-app (run, report, list, diff, validate)"
```

---

## Session 4: Author Initial Tasks

### Task 13: code-gen tasks (4 tasks)

Create `benchmarks/tasks/code-gen/{001-fibonacci,002-fizzbuzz,003-lru-cache-class,004-csv-parser}/` each with `task.yaml`, `seed/`, `eval/check.sh`.

**Commit per batch:**
```bash
git commit -m "bench(tasks): add 4 code-gen tasks (fibonacci, fizzbuzz, lru-cache, csv-parser)"
```

### Task 14: bug-fix tasks (3 tasks)

Create `benchmarks/tasks/bug-fix/{001-off-by-one,002-race-condition,003-broad-except-swallow}/`.

```bash
git commit -m "bench(tasks): add 3 bug-fix tasks"
```

### Task 15: refactor + test-gen tasks (4 tasks)

Create `benchmarks/tasks/refactor/{001-long-function-split,002-god-class-decompose}/` and `benchmarks/tasks/test-gen/{001-no-tests-coverage-80,002-edge-cases-existing-module}/`.

```bash
git commit -m "bench(tasks): add 2 refactor + 2 test-gen tasks"
```

### Task 16: exploration + ml-tools + long-horizon tasks (5 tasks)

Create remaining tasks:
- `exploration/{001-where-handled,002-explain-data-flow}/`
- `ml-tools/{001-inspect-csv-polars,002-write-model-card}/`
- `long-horizon/{001-rest-endpoint-end-to-end}/`

```bash
git commit -m "bench(tasks): add exploration, ml-tools, and long-horizon tasks (5)"
```

---

## Session 5: Integration, Baseline Run, Polish

### Task 17: bench validate — ensure all 16 tasks pass validation

Run: `uv run norn bench validate`
Expected: 16/16 valid, 0 errors.

```bash
git commit -m "fix(bench): fix any task validation issues found during validate pass"
```

### Task 18: Full test suite green

Run: `uv run pytest -x -q`
Expected: ~590+ tests pass (560 existing + ~33 new bench runner tests).

### Task 19: Baseline run (manual, not committed)

Run: `uv run norn bench run`
This will actually invoke the LLM. Review results with `uv run norn bench report`.
Save report to `benchmarks/reports/2026-04-22-baseline.md`.

```bash
git commit -m "bench(reports): add initial baseline report"
```

### Task 20: Push to Forgejo

```bash
git push forgejo main
```

---

## Summary

| Session | Tasks | Tests added | Key deliverable |
|---------|-------|-------------|-----------------|
| 1 | 1–4 | ~10 | Config + models + sandbox |
| 2 | 5–8 | ~16 | Executor + evaluator + judge + metrics |
| 3 | 9–12 | ~12 | CLI sub-app + persistence + reporter |
| 4 | 13–16 | 0 (task content) | 16 benchmark tasks authored |
| 5 | 17–20 | ~2 (integration) | Validation, baseline run, push |
| **Total** | **20 tasks** | **~40 tests** | **Full benchmark system** |

---

## Files Created (new)

```
benchmarks/
├── __init__.py
├── runner/
│   ├── __init__.py
│   ├── models.py
│   ├── sandbox.py
│   ├── metrics.py
│   ├── executor.py
│   ├── evaluator.py
│   ├── judge.py
│   ├── persistence.py
│   ├── reporter.py
│   └── loader.py
├── tasks/             (16 task directories, each with task.yaml + seed/)
├── results/           (gitignored except README.md)
└── reports/
src/norn/cli/bench.py
tests/test_bench/
├── __init__.py
├── test_models.py
├── test_sandbox.py
├── test_metrics.py
├── test_executor.py
├── test_evaluator.py
├── test_judge.py
├── test_persistence.py
├── test_reporter.py
├── test_loader.py
└── test_bench_cli.py
```

## Files Modified (existing)

```
src/norn/core/config.py          — BenchConfig, JudgeConfig, NORN_CONFIG_OVERRIDES
src/norn/cli/main.py             — register bench sub-app
configs/default.yaml             — bench section
.gitignore                       — benchmarks/results/*.jsonl
```
