# Norn SOTA v2 — Wave 2 Design (2026-08-29)

## Context

Wave 1 shipped and validated (16/16 internal, `benchmarks/reports/2026-08-28-sota-v2-wave1-validation.md`). First external run on terminal-bench 2: **32/89 = 36%**, a clean number after all 18 infra artefacts were fixed (`benchmarks/reports/2026-08-28-terminal-bench-2-harbor.md`). Failures cluster on **long-horizon / compute-heavy / multi-step** tasks — the agent gets lost across long sequences; the 9 timeouts fail even at 2× time budget, so it's a task-management gap, not a speed gap.

Wave 2 targets that gap on three axes chosen by the owner: **long-horizon**, **AXI tool-output**, **plan/decompose**. Same hard constraint as wave 1: **additive only**, new flags off by default, validated on the 89 tb2 tasks via `NORN_CONFIG_OVERRIDES` (zero default flips → zero regression exposure for interactive users) before any default is flipped.

Two structural facts (verified) make all of this safe:
- `AgentLoop.history` is strictly append-only; compaction/injection happens on the *view* sent to the LLM, not the ledger. `ThreadLedger.verify_derivation` already exempts `Role.SYSTEM` messages, so summary/notes injection passes strict CI (`NORN_STRICT_INVARIANTS=1`).
- Prompt cache marks only `messages[0]` (system prefix) + last tool schema. Injecting extra SYSTEM messages after the prefix, or appending USER/ASSISTANT turns, never touches a cache breakpoint.

---

## Axis A — Long-horizon (highest ROI)

### A1. Persistent task scratchpad (do first)
Direct fix for "agent loses the thread across many steps": a durable plan/progress file that survives context compaction.
- **New `src/norn/core/task_notes.py`** — `TaskNotes(path, max_chars=4000)`: `read()`, `write_section(section, content)` (overwrite per section, not append → bounded), `render()` (clamped, evicts oldest Progress lines first). Fixed sections: Goal / Plan-TODO / Decisions / Progress.
- **New `src/norn/tools/task_notes_tool.py`** — `TaskNotesTool`, `input_model {action: Literal["read","set_plan","note_progress","record_decision"], content=""}`, risk LOW; delegates to injected `TaskNotes`.
- **agent.py**: ctor param `task_notes: TaskNotes | None`. In `_run_impl`/`run_stream`, inject `Message(role=SYSTEM, content=task_notes.render())` right after the system message (same slot as the summary) when non-empty → cache-neutral, ledger-exempt, survives window eviction (regenerated from file).
- **config.py `AgentConfig`**: `task_notes: bool = False`, `task_notes_path: str = ".norn/task_notes.md"`, `task_notes_max_chars: int = 4000`.
- Wire in `cli/main.py` (registry + both AgentLoop sites) and `cli/bench.py` `_build_config_overrides`. One line in `AGENT_SYSTEM_PROMPT` telling the agent to keep the plan updated on multi-step tasks (harmless, can be unconditional).
- Tests: section overwrite, render clamp/eviction, injected msg is SYSTEM & positioned after prefix, `verify_derivation` clean under strict, tool round-trips.

### A2. Sliding window — validate, then staged default-on
The compaction partner to the scratchpad (window compacts raw turns; scratchpad keeps the distilled plan). Already ledger-safe and cache-safe; only fires when history > `max_history_tokens`, so short tasks (incl. most of the 32 passing) never trigger it.
- **Do NOT flip the default yet.** Enable via bench `_build_config_overrides`: `{"context": {"sliding_window": True}}`, keep the **extractive** summary (no `summary_model`) — free, deterministic, and sidesteps the per-turn re-summarization cost of the LLM path (`_get_summary` caches on `history_len`). Validate ≥32 on 89, then flip `ContextConfig.sliding_window` default in a follow-up.
- Optional (low priority): replace `_estimate_tokens` chars/4 with `litellm.token_counter` behind try/except.
- Tests: window triggers only above threshold; strict `verify_derivation` clean with `[system, summary, recent...]`; extractive summary stable across calls; agent run >window turns under strict.

### A3. Adaptive turn budget (minimal, validate-or-drop)
Timeouts fail at 2× time → not a time problem; extension only helps tasks needing >50 *productive* rounds. Keep it minimal and reuse `_detect_failure_patterns`.
- **agent.py**: `_recent_progress(messages, window=K) -> bool` (successful file_write/edit or non-error bash in last K rounds). At `max_tool_rounds`, if progress AND no `error_flood`/`tool_loop`, grant one bounded extension (+50%, `max_round_extensions=1`); the negative gate also cuts stuck tasks early (saves time). Apply in both `run` and `run_stream`.
- **config.py**: `adaptive_rounds: bool = False`, `max_round_extensions: int = 1`, `round_extension_factor: float = 0.5`.
- Tests: extension granted on progress, refused under error_flood/tool_loop, hard ceiling respected, `[Max tool rounds reached]` still emitted when exhausted.

---

## Axis B — AXI tool-output (transversal cost/precision)

Reduce wasted exploration turns: aggregates, truncation signals, next-step hints on tool outputs. Additive, `output` stays a `str`, default path byte-identical when off.
- **New `src/norn/tools/axi_format.py`** — pure helpers, no tool imports (no cycles): `axi_list_result(items, total, shown, noun, full_hint, empty_hint)`, `axi_match_result(lines, total_matches, file_count, truncated, full_hint)`, `axi_file_header(path, start, end, total_lines)`, `axi_empty(noun, next_step)`. Aggregate/next-step line **first** so truncation (bookend `head`) keeps it.
- **Flag** `axi_output` (config.FlagsConfig + `_build_flag_registry` + `NORN_FLAG_AXI_OUTPUT`), off by default.
- **Prototype 3 hot-path tools** (order): `grep_tool.py` (`N matches in M files`, truncation notice at the 200 cap, empty→broaden hint), `glob_tool.py` (`N files (showing min(N,500))`, 500-cap notice), `file_read.py` (dir: `N entries (D dirs, F files)`; file: `lines A-B of TOTAL` header → agent never re-reads to learn length). Additive ctor `__init__(self, *, axi_output=False)`, branch inside `execute`, current return stays as the `else`.
- Wire ctor kwargs in `cli/main.py:_build_registry` (pattern already used for `BashTool(sandbox_enabled=...)`).
- Tests: new `tests/test_tools/test_axi_format.py` (aggregate math, truncation/empty templates) + per-tool cases with `axi_output=True`. Leave existing default-off tests untouched (they guard no-regression).
- Deferred to later PRs (same helper): bash STDERR/exit summary, coderag_* result headers.

---

## Axis C — Plan/decompose (auto_plan)

The gap: `/plan` is interactive only; headless runs (bench, `norn run`) get no pre-task decomposition. Mirror `auto_verify` but pre-loop, folded into the existing `run_verified` (no parallel method, no verify-loop duplication).
- **prompts.py**: add `PLAN_PROMPT` (restate task+acceptance, numbered steps, files, test strategy; explicit "do NOT modify files yet"; **no PASS/FAIL token** so `_extract_verdict` never mis-reads it). Optional one-line append to `SELF_VERIFY_PROMPT`: "[ ] Every numbered step of your plan is [DONE]" to close decomposition↔verification.
- **agent.py `run_verified`**: additive kwargs `plan_first=False`, `plan_prompt=None`, `plan_min_chars=0`. When `plan_first` and `len(user_input) >= plan_min_chars`, run one `run_stream(PLAN_PROMPT + task)` turn before the main turn. Plan lives in history (message append, not system change → cache-safe, ledger-safe). `max_verify_rounds=0` already degrades to plain run, so `auto_plan` works without `auto_verify`.
- **config.py `AgentConfig`**: `auto_plan: bool = False`, `auto_plan_min_chars: int = 0` (0 = always plan when on; bench starts at 0 to measure, tune ~250-300 later).
- **cli/main.py** (~l.556): branch into `run_verified` when `auto_verify OR auto_plan`, passing `plan_first`/`plan_min_chars`; `max_verify_rounds=0` when auto_verify off.
- **harbor_adapter/norn_agent.py**: add `"auto_plan": True` to `_CONFIG_OVERRIDES["agent"]`.
- Tests: plan-then-task (2 turns), skip below threshold, plan+verify (3 turns), no-plan-by-default regression, `PLAN_PROMPT` verdict-free, config defaults.
- Risk: plan turn implementing early (yolo) → mitigated by explicit "do NOT modify files yet", monitor on bench; extra input tokens (task text twice) — minor, opt-in.

---

## Execution order & validation

1. **A1 task scratchpad** (new files, lowest core-loop risk) → 2. **C auto_plan** (small, reuses run_verified) → 3. **B AXI** (grep→glob→file_read) → 4. **A2 sliding window** (bench-enable) → 5. **A3 adaptive rounds** (validate-or-drop).

Each lands as its own commit(s), full suite green under strict invariants between each.

**Bench-driven validation:** after each axis, re-run the relevant tb2 subset (start with the ~15 long-horizon/exploration failures, not the full 89) with the axis enabled via overrides. Flip a default only after a clean full-89 run shows ≥32 with the axis on. Harbor adapter accumulates the enabled flags for a final full re-run.

## Verification
- `uv run pytest` (all new tests + suite under `NORN_STRICT_INVARIANTS=1`), `uv run ruff check .` (keep baseline flat).
- Internal bench `norn bench run` unaffected with defaults off.
- tb2 subset re-runs per axis via the Harbor adapter; final full-89 re-run with the wave-2 flags on → compare to the 32/89 baseline.
