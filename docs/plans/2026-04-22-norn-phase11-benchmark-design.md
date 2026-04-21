# Norn — Phase 11 Design : Internal Mini-Eval Benchmark

**Date:** 2026-04-22
**Status:** Approved, ready for implementation planning
**Branch:** `chore/planning-future-phases`
**Prerequisite:** Phase 9 v2 G (prompt caching) — for `cache_hit_rate` metric

---

## 1. Goal

Measure objectively how Norn evolves between versions (prompt changes,
model swaps, refactors, new tools) by running a curated suite of 16–23
tasks across 7 categories and comparing results.

The benchmark is a **diagnostic tool**, not a leaderboard. Primary
audience: the project owner, before merging non-trivial changes.

Two-tier strategy:

1. **Mini-eval custom MLOps** (this phase) — 16+ tasks specific to Norn's
   workflows, light enough to run locally on demand.
2. **SWE-bench Lite integration** (deferred to Phase 11.5+) — industry
   standard for cross-comparison; significant infrastructure.

---

## 2. Out of scope

- SWE-bench Lite (deferred to dedicated phase)
- CI integration (LLM-in-CI = secrets + cost + flakiness; manual local only)
- Parallel task execution (sequential MVP)
- Web UI / dashboard (Rich CLI sufficient)
- Automated cross-model comparison (manual: run twice with
  `--model fast` vs `--model powerful`, then `bench diff`)
- Multi-language tasks (Python only)
- Cost-in-USD tracking (tokens + cache hits cover the efficiency
  signal without depending on a USD-mapping table that varies per model)

---

## 3. Architecture

```
benchmarks/
├── tasks/                        # versioned task definitions
│   ├── code-gen/
│   │   ├── 001-fibonacci/
│   │   │   ├── task.yaml         # prompt, criteria, timeout, n_runs
│   │   │   ├── seed/             # initial repo state
│   │   │   │   ├── pyproject.toml
│   │   │   │   ├── src/__init__.py
│   │   │   │   └── tests/test_solution.py
│   │   │   └── eval/
│   │   │       └── check.sh      # validation command
│   │   └── 002-...
│   ├── bug-fix/
│   ├── refactor/
│   ├── test-gen/
│   ├── exploration/
│   ├── ml-tools/
│   └── long-horizon/
├── runner/                       # Python runner (importable module)
│   ├── __init__.py
│   ├── models.py                 # TaskDef, TaskResult, RunReport, ExecutionTrace
│   ├── sandbox.py                # tmpdir + git checkout
│   ├── executor.py               # invokes Norn against task
│   ├── evaluator.py              # pass/fail + LLM-as-judge
│   ├── metrics.py                # extract metrics from observability events
│   ├── reporter.py               # markdown report + comparison
│   └── judge.py                  # LLM-as-judge wrapper
├── results/                      # JSONL append-only (gitignored except summaries)
│   ├── 2026-04-22T15-30-12_3156259.jsonl
│   ├── 2026-04-23T09-15-44_5a4da3b.jsonl
│   └── README.md
└── reports/                      # versioned markdown reports
    ├── 2026-04-22-baseline.md
    └── 2026-04-23-after-phase9-v2.md
```

The runner lives under `benchmarks/runner/` rather than `src/norn/bench/`
to keep it out of the published package. The CLI sub-app
(`norn bench ...`) lives in `src/norn/cli/bench.py` and imports from
`benchmarks.runner`.

---

## 4. Task definition format

`benchmarks/tasks/<category>/<id>/task.yaml`:

```yaml
id: "code-gen-001-fibonacci"
category: "code-gen"
description: "Implement an iterative Fibonacci function"
prompt: |
  In the file src/solution.py, implement a function `fib(n: int) -> int`
  that returns the n-th Fibonacci number (0-indexed: fib(0)=0, fib(1)=1).
  Use an iterative approach. All tests in tests/test_solution.py must pass.
seed_dir: "seed/"
eval_command: "uv run pytest tests/ -v"
expected_files_changed: ["src/solution.py"]
forbidden_files_changed: []   # optional: fail if these are modified
timeout_seconds: 300
n_runs: 3
metadata:
  difficulty: "easy"
  expected_turns: 2
  uses_tools: ["file_read", "file_write"]
  judge_required: false       # true for refactor/exploration categories
```

Validation via Pydantic `TaskDef` model. Loader walks
`benchmarks/tasks/**/task.yaml` and instantiates one `TaskDef` per file.

---

## 5. Sandbox isolation

```python
def create_sandbox(task: TaskDef) -> Path:
    """Copy seed/ to a tmpdir, init git, return path.

    Each task runs in a fresh isolated directory. Norn is launched with
    cwd=sandbox so its tools cannot escape (combined with permission
    mode 'strict' for benchmark runs).
    """
    tmpdir = Path(tempfile.mkdtemp(prefix=f"norn-bench-{task.id}-"))
    shutil.copytree(task.seed_dir, tmpdir, dirs_exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=tmpdir, check=True)
    subprocess.run(["git", "add", "-A"], cwd=tmpdir, check=True)
    subprocess.run(
        ["git", "-c", "user.email=bench@norn", "-c", "user.name=Bench",
         "commit", "-qm", "seed"],
        cwd=tmpdir, check=True,
    )
    return tmpdir


def cleanup_sandbox(path: Path) -> None:
    """Remove the sandbox unless `--keep-sandbox` was passed."""
    shutil.rmtree(path, ignore_errors=True)
```

Sandboxes are kept on failure when `--keep-sandbox` is passed (debugging).
Otherwise removed after eval to bound disk usage.

---

## 6. Executor

```python
async def execute_task(
    task: TaskDef,
    sandbox: Path,
    config_overrides: dict,
) -> ExecutionTrace:
    """Run Norn on task.prompt with cwd=sandbox.

    Captures observability events to a per-task log file and parses
    them after completion. Enforces task.timeout_seconds.
    """
    log_path = sandbox / ".norn-bench-log.jsonl"
    config_overrides = {
        **config_overrides,
        "logging.file_dir": str(sandbox),
        "logging.output": "file",
        "permissions.mode": "strict",   # block destructive ops in bench
    }
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        "norn", "run", task.prompt,
        cwd=sandbox,
        env={**os.environ, "NORN_CONFIG_OVERRIDES": json.dumps(config_overrides)},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=task.timeout_seconds,
        )
    except asyncio.TimeoutError:
        proc.kill()
        return ExecutionTrace.timeout(task.timeout_seconds)

    duration_ms = (time.monotonic() - start) * 1000
    events = parse_jsonl(log_path) if log_path.exists() else []
    return ExecutionTrace(
        events=events,
        duration_ms=duration_ms,
        stdout=stdout.decode(),
        stderr=stderr.decode(),
        files_changed=git_diff_name_only(sandbox),
        timed_out=False,
    )
```

(The `NORN_CONFIG_OVERRIDES` env var injection requires a small addition
to `NornConfig.apply_env_overrides` — flagged as an open question if it
doesn't already exist.)

---

## 7. Evaluator

```python
def evaluate_task(
    task: TaskDef,
    trace: ExecutionTrace,
    sandbox: Path,
    judge: Judge | None,
) -> TaskResult:
    """Run task.eval_command, capture pass/fail + extract metrics."""
    if trace.timed_out:
        return TaskResult.timeout(task, trace)

    eval_proc = subprocess.run(
        task.eval_command, shell=True, cwd=sandbox,
        capture_output=True, timeout=120,
    )
    binary_pass = eval_proc.returncode == 0

    judge_score: JudgeScore | None = None
    if task.metadata.get("judge_required") and judge is not None:
        judge_score = judge.evaluate(task, trace, sandbox)

    return TaskResult(
        task_id=task.id,
        category=task.category,
        success=binary_pass and (judge_score is None or judge_score.passes()),
        binary_pass=binary_pass,
        judge_score=judge_score,
        # Agent metrics extracted from observability events
        total_turns=count_events(trace.events, "llm.complete"),
        total_tokens=sum_field(trace.events, "total_tokens"),
        prompt_tokens=sum_field(trace.events, "prompt_tokens"),
        completion_tokens=sum_field(trace.events, "completion_tokens"),
        cache_read_tokens=sum_field(trace.events, "cache_read_tokens"),
        cache_creation_tokens=sum_field(trace.events, "cache_creation_tokens"),
        latency_ms=trace.duration_ms,
        tool_calls_distribution=count_by_tool(trace.events),
        permission_denials=count_events_by(
            trace.events, "tool.call",
            lambda e: e.get("error_type") == "PermissionDenied",
        ),
        # Eval result
        eval_stdout=eval_proc.stdout.decode()[:4000],   # truncate
        eval_returncode=eval_proc.returncode,
        files_changed=trace.files_changed,
    )
```

`cache_hit_rate` is derived at aggregation time as
`cache_read_tokens / prompt_tokens` (depends on Phase 9 v2 G being
landed).

---

## 8. LLM-as-judge

For non-binary categories (refactor, exploration, sometimes long-horizon):

```python
JUDGE_PROMPT = """You evaluate an AI agent's response to a task. Be strict.

Task: {task_description}
Expected outcome: {expected_outcome}
Original code (if relevant):
{original_code}

Agent's changes (git diff):
{git_diff}

Agent's final answer (if any):
{agent_answer}

Score on three axes (0-10 each):
1. Correctness — does it solve the task?
2. Quality — is the code/answer clean and idiomatic?
3. Completeness — is anything missing?

Return JSON only:
{{"correctness": N, "quality": N, "completeness": N, "reasoning": "..."}}
"""


class Judge:
    def __init__(self, model: str, llm_provider: LLMProvider):
        self.model = model
        self.llm = llm_provider

    async def evaluate(
        self, task: TaskDef, trace: ExecutionTrace, sandbox: Path,
    ) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(...)
        response = await self.llm.complete(
            messages=[Message(role=Role.USER, content=prompt)],
            temperature=0.0,
        )
        return JudgeScore.from_json(response.content)
```

Defaults to a model **independent from the one being benchmarked** to
avoid self-evaluation bias (e.g. benchmark `qwen3-coder` with judge
`anthropic/claude-sonnet-4`). Pass threshold: average score ≥ 7/10
across all three axes.

---

## 9. CLI commands

| Command | Behaviour |
|---|---|
| `norn bench run` | Run all tasks. Options: `--category <name>`, `--task <id>`, `--n-runs N`, `--model <tier>`, `--keep-sandbox`, `--output <path>`. |
| `norn bench report` | Show last run's report (Rich table). Options: `--baseline <run_file>` to compare, `--format markdown` to emit md, `--category <name>` to filter. |
| `norn bench list` | List available tasks with category, difficulty, last execution outcome. |
| `norn bench diff <run_a> <run_b>` | Compare two runs side-by-side; highlight regressions (success → fail), token bloat (>20% increase), latency regressions. |
| `norn bench validate` | Lint `task.yaml` files: schema valid, seed/ exists, eval/check.sh exists, etc. Useful before committing a new task. |

All CLI commands respect `flags.bench: false` default — disabled until
explicitly enabled in config.

---

## 10. Configuration

Additions to `configs/default.yaml`:

```yaml
flags:
  bench: false              # opt-in, no extras required (uses existing deps)

bench:
  tasks_dir: "benchmarks/tasks"
  results_dir: "benchmarks/results"
  reports_dir: "benchmarks/reports"
  default_n_runs: 1
  default_timeout_seconds: 300
  judge:
    enabled: true
    model: "anthropic/claude-sonnet-4"   # independent from main model
    pass_threshold: 7.0
  parallel_tasks: 1         # sequential MVP
```

A new Pydantic `BenchConfig` block in `NornConfig`.

---

## 11. Metrics tracked

### Per task (per run)

| Metric | Source |
|---|---|
| `success` | binary pass AND judge pass (if applicable) |
| `binary_pass` | `eval_command.returncode == 0` |
| `judge_score` | LLM judge (only if `judge_required`) |
| `total_turns` | count of `llm.complete` events |
| `prompt_tokens`, `completion_tokens`, `total_tokens` | sum from `llm.complete` |
| `cache_read_tokens`, `cache_creation_tokens` | sum from `llm.complete` (Phase 9 v2 G) |
| `latency_ms` | end-to-end wall clock |
| `tool_calls_distribution` | `{tool_name: count}` from `tool.call` |
| `permission_denials` | count of `tool.call` with `error_type=PermissionDenied` |
| `files_changed` | `git diff --name-only` of sandbox |
| `eval_stdout` (truncated) | for debugging |
| `timed_out` | bool |

### Aggregated per run

| Metric | Compute |
|---|---|
| Overall success rate | `passed / total` |
| Per-category success rate | grouped |
| Total tokens / run | sum |
| Total cache hits | `sum(cache_read) / sum(prompt)` |
| Avg turns / task | mean |
| Avg latency / task | mean |
| Avg permission denials / task | mean |
| Variance (when `n_runs > 1`) | std dev of binary_pass per task |

---

## 12. JSONL output format

`benchmarks/results/<ts>_<git_sha>.jsonl`:

```jsonl
{"type": "run_meta", "timestamp": "2026-04-22T15:30:12Z", "git_sha": "3156259", "norn_version": "0.1.0", "model": "anthropic/claude-sonnet-4", "n_runs": 3, "n_tasks": 18}
{"type": "task_result", "task_id": "code-gen-001-fibonacci", "run_idx": 1, "success": true, "binary_pass": true, "total_turns": 2, "total_tokens": 1340, "cache_read_tokens": 800, "latency_ms": 4521, ...}
{"type": "task_result", "task_id": "code-gen-001-fibonacci", "run_idx": 2, "success": true, ...}
{"type": "task_result", "task_id": "code-gen-001-fibonacci", "run_idx": 3, "success": true, ...}
{"type": "task_result", "task_id": "bug-fix-001-off-by-one", ...}
{"type": "run_summary", "overall_success_rate": 0.87, "by_category": {"code-gen": 1.0, "bug-fix": 0.67, ...}, "total_tokens": 24530, "total_cache_read_tokens": 12450, ...}
```

Append-only. Each `bench run` produces exactly one file.

---

## 13. Initial task inventory (16 tasks)

| Category | Count | Tasks |
|---|---|---|
| code-gen | 4 | 001-fibonacci, 002-fizzbuzz, 003-lru-cache-class, 004-csv-parser |
| bug-fix | 3 | 001-off-by-one, 002-race-condition, 003-broad-except-swallow |
| refactor | 2 | 001-long-function-split, 002-god-class-decompose |
| test-gen | 2 | 001-no-tests-coverage-80, 002-edge-cases-existing-module |
| exploration | 2 | 001-where-handled, 002-explain-data-flow |
| ml-tools | 2 | 001-inspect-csv-polars, 002-write-model-card |
| long-horizon | 1 | 001-rest-endpoint-end-to-end |

Each task ships with: `task.yaml`, `seed/` (with `pyproject.toml` + tests
where applicable), `eval/check.sh`. Total seed footprint expected:
~150 KB for all 16 tasks combined.

---

## 14. Tests (of the runner itself)

| Layer | Tests | Approach |
|---|---|---|
| TaskDef | ~5 | YAML parse, Pydantic validation, missing fields, malformed timeout, default n_runs |
| Sandbox | ~3 | create + cleanup, git init, isolation (writes don't escape) |
| Executor | ~4 | mock norn run via fake subprocess, timeout handling, log parsing, env override propagation |
| Evaluator | ~5 | pass extraction, fail extraction, judge integration (mocked), metric extraction from synthetic events, files_changed tracking |
| Judge | ~3 | JSON parsing, malformed response handling, threshold logic |
| Metrics | ~4 | aggregation correctness, cache_hit_rate computation, distribution counting |
| Reporter | ~4 | JSONL writes, markdown rendering, baseline comparison, regression detection |
| CLI | ~5 | smoke tests for `bench run`, `bench report`, `bench list`, `bench diff`, `bench validate` |

Total: **~33 tests** for the runner. Plus 16 task definitions (each with
its own `seed/` and `eval/check.sh` — those are not pytest tests but
exercised end-to-end by `bench run`).

---

## 15. Risks & mitigations

| Risk | Mitigation |
|---|---|
| LLM stochasticity makes "stable" tasks fail intermittently | `n_runs >= 3` default for non-deterministic tasks, report variance, eyeball failures before raising regression alarm |
| Judge LLM biased toward certain styles | Independent model, strict JSON-only prompt, threshold ≥ 7/10, judge optional per-task |
| Tasks become obsolete as Norn evolves | Tasks versioned in git; deprecate explicitly via `task.yaml: deprecated: true` (skipped by `bench run`) |
| Cost explosion (`16 tasks × 3 runs × 5k tokens`) | Default `n_runs=1`, default model is the configured one (often free tier), token dashboard in report |
| Sandbox leak (Norn writes outside cwd) | Permission `strict` mode for bench, sandbox is tmpdir, cleanup after run |
| Eval deterministic but output non-deterministic | Tests assert on signature/behaviour, not exact formatting |
| Long-horizon task hangs | Timeout enforced (default 300 s, override per task), `max_turns` cap in Norn (existing) |
| `NORN_CONFIG_OVERRIDES` env var doesn't exist | Open question — add it as a 1-line change in `NornConfig.apply_env_overrides` if missing |
| Bench run pollutes the user's `~/.norn/logs` | Override `logging.file_dir` to sandbox per-task |

---

## 16. Effort estimate

5 sessions:

1. **Session 1** — Models + sandbox + executor (TDD)
2. **Session 2** — Evaluator + Judge + metrics aggregation
3. **Session 3** — CLI sub-app + JSONL persistence + reporter
4. **Session 4** — Author the 16 initial tasks (this is the longest in
   wall-clock time, but each task is small)
5. **Session 5** — Markdown report generator + baseline comparison +
   `bench diff` + docs

---

## 17. Success criteria

- `norn bench run --category code-gen` runs the 4 code-gen tasks without
  error and writes a JSONL result file
- `norn bench report` renders a Rich table with per-category success
  rate and aggregate metrics
- `norn bench diff <run_a> <run_b>` clearly highlights at least one
  regression (success → fail) on a deliberately-broken commit
- 16 initial tasks land and pass at ≥ 70% on the baseline model
  (`anthropic/claude-sonnet-4`)
- Runner adds ~33 tests; total suite reaches ~600+ on `main`
- Full bench (16 tasks × 1 run) completes in < 30 minutes on baseline
- `norn bench validate` catches malformed task definitions before commit

---

## 18. Open questions

1. **`NORN_CONFIG_OVERRIDES` env var** — does Norn's config loader
   already support env-based overrides for nested keys
   (`logging.file_dir`)? If not, add a tiny adapter as part of
   Session 1.
2. **Strict permission mode in bench** — `strict` currently denies
   destructive without a handler. Need to verify it doesn't block
   legitimate `file_write` for code-gen tasks; might need a
   bench-specific permission profile.
3. **Reproducibility seed** — should we set a fixed seed on the LLM
   call for deterministic reproduction? `temperature=0.0` is the closest
   we get; OpenAI exposes `seed=` but it's not honoured everywhere.
   Document the limitation.
4. **Tasks with network access** — none of the initial 16 require
   network. Future tasks (web scraping, API clients) will. Decide on a
   per-task `network_required: true` flag to opt out of an offline-only
   policy.
5. **Cross-model comparison ergonomics** — `bench diff` currently
   compares two specific runs. Add a future `bench compare-models
   <model_a> <model_b>` that runs both back-to-back and produces a
   single comparison report.
6. **Storing seeds in git** — 16 tasks × small seed directories should
   be < 200 KB. Acceptable to commit. If it grows beyond ~5 MB, switch
   to `git lfs` or fetch on first use.

---

## 19. Approval

Design approved by user (FR session, 2026-04-22). Branch:
`chore/planning-future-phases` (sibling worktree at
`~/norn-planning`).

Next step: continue planning batch with Phase 12 (`watch` mode design)
and OSS hygiene (LICENSE + CONTRIBUTING + templates) before any
implementation. Implementation plan deferred to a later iteration.
