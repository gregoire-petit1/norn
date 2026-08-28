"""Benchmark data models for the norn benchmark runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class TaskMetadata(BaseModel):
    """Optional metadata attached to a task definition."""

    tags: list[str] = Field(default_factory=list)
    difficulty: str | None = None
    source: str | None = None


class TaskDef(BaseModel):
    """Task definition loaded from a YAML file."""

    id: str
    category: str
    description: str
    prompt: str
    seed_dir: str
    eval_command: str
    timeout_seconds: int = 120
    n_runs: int = 1
    metadata: TaskMetadata = Field(default_factory=TaskMetadata)

    # Internal: path to the YAML file this was loaded from
    _source_path: Path | None = None

    @classmethod
    def from_yaml(cls, path: Path | str) -> TaskDef:
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        td = cls.model_validate(data)
        td._source_path = path
        return td

    @property
    def absolute_seed_dir(self) -> Path:
        if self._source_path is not None:
            return self._source_path.parent / self.seed_dir
        return Path(self.seed_dir)


class JudgeScore(BaseModel):
    """Score assigned by an LLM judge."""

    correctness: float
    quality: float
    completeness: float
    reasoning: str

    def average(self) -> float:
        return (self.correctness + self.quality + self.completeness) / 3.0

    def passes(self, threshold: float = 7.0) -> bool:
        return self.average() >= threshold


class ExecutionTrace(BaseModel):
    """Trace of a single task execution."""

    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: int = 0
    timed_out: bool = False
    eval_exit_code: int | None = None
    eval_stdout: str = ""
    eval_stderr: str = ""

    @classmethod
    def timeout(cls, timeout_seconds: int) -> ExecutionTrace:
        return cls(
            timed_out=True,
            duration_ms=timeout_seconds * 1000,
        )


class TaskResult(BaseModel):
    """Result of running a single task."""

    task_id: str
    success: bool = False
    binary_pass: bool = False
    timed_out: bool = False
    latency_ms: int = 0
    trace: ExecutionTrace = Field(default_factory=ExecutionTrace)
    judge_score: JudgeScore | None = None
    eval_returncode: int = -1
    eval_stdout: str = ""
    error: str | None = None
    # SOTA v2 (workstream B): per-task session metrics, extracted best-effort
    # from the JSONL observability logs. Empty when unavailable (e.g. Docker).
    tool_counts: dict[str, int] = Field(default_factory=dict)
    llm_calls: int = 0

    @classmethod
    def timeout(cls, task: TaskDef, trace: ExecutionTrace) -> TaskResult:
        return cls(
            task_id=task.id,
            success=False,
            timed_out=True,
            latency_ms=trace.duration_ms,
            trace=trace,
            error="Execution timed out",
        )


class RunMeta(BaseModel):
    """Metadata for a benchmark run."""

    run_id: str = ""
    model: str = ""
    agent_version: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    config: dict[str, Any] = Field(default_factory=dict)


class RunReport(BaseModel):
    """Aggregate report for a full benchmark run."""

    meta: RunMeta = Field(default_factory=RunMeta)
    results: list[TaskResult] = Field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed(self) -> int:
        return self.total - self.passed

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0
