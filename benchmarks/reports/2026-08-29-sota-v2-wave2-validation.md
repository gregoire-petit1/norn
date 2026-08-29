# SOTA v2 Wave 2 — Validation Report (TEMPLATE — awaiting tb2 run)

> **Status: PENDING.** Code complete and merged; the terminal-bench-2 A/B is
> blocked on Copilot quota until **2026-09-01** (the `premium_interactions`
> bucket that `claude-sonnet-4.5` API calls consume is at 0%). Fill the
> `<...>` slots after the run. Baseline to beat: **32/89 (36.0%)**
> (`2026-08-28-terminal-bench-2-harbor.md`).

## What wave 2 shipped

Five additive long-horizon / cost axes, all flags off by default, invariant
append-only kept strict-green across the suite (1014+ tests). Design:
`docs/plans/2026-08-29-norn-sota-v2-wave2.md`.

| Axis | Flag | Commit | What it does |
|---|---|---|---|
| A1 task scratchpad | `agent.task_notes` | `654646f` | Durable Goal/Plan/Decisions/Progress file, injected as a SYSTEM message re-rendered each round; survives compaction |
| C auto_plan | `agent.auto_plan` | `d97c874` | Pre-task decomposition turn (PLAN_PROMPT) folded into run_verified |
| B AXI output | `flags.axi_output` | `3699267` | Aggregates + truncation signal + next-step hints + line-range on grep/glob/file_read |
| A3 adaptive rounds | `agent.adaptive_rounds` | `9bb95d2` | One progress-gated round-budget extension; anti-stuck negative gate |
| A2 sliding window | `context.sliding_window` | (pre-existing) | Compacts long histories; extractive summary (free, deterministic) |

Harbor A/B switch: `NORN_WAVE2=1` (`bd2e051`) merges all five into the
adapter overrides; unset = wave-1 baseline. CLI-wiring guard test added after
a kwarg mis-routing bug (`a92ed0f`).

## How to run (after 2026-09-01)

Confirm quota first:
```
curl -s -H "Authorization: token $(cat ~/.config/litellm/github_copilot/access-token)" \
  -H "Editor-Version: vscode/1.90" https://api.github.com/copilot_internal/user \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['quota_snapshots']['premium_interactions'])"
```
`percent_remaining` must be > 0. Then:
```
NORN_WAVE2=1 PYTHONPATH=. <harbor-venv>/bin/harbor run \
  -d terminal-bench/terminal-bench-2 \
  -a benchmarks.harbor_adapter.norn_agent:NornAgent \
  --agent-timeout-multiplier 1.5 \
  -o <jobs-dir> --job-name tb2-wave2 --force-build -n 4 -q -y
```

## Headline (fill in)

**Wave 2 reward mean: `<X.XXX>` — `<N>`/89.**  vs baseline 0.360 — 32/89.
Delta: **`<+/-K>` tasks (`<+/-P>` pts)**.

## Breakdown (fill in)

| Outcome | Wave 2 | Baseline |
|---|---|---|
| Passed | `<N>` | 32 |
| Failed (ran) | `<N>` | 49 |
| Agent timeout | `<N>` | 0 (was 9, fixed via 2× in baseline re-run) |
| Install-failed | `<N>` (expect 0) | 0 |

## Per-task deltas (fill in)

- **Newly passing** (fail→pass under wave 2): `<list>`
- **Newly failing** (pass→fail — regressions to investigate): `<list>`
- **Still failing** (both): `<count>`

## Per-axis read (fill in from agent logs)

For a sample of newly-passing and still-failing trials, inspect
`<job>/<task>/agent/norn.txt`:
- **task_notes** — did the agent set a plan and note progress? Did it stay
  oriented on long tasks? `<obs>`
- **auto_plan** — did the plan turn produce a usable decomposition without
  implementing early (the "do NOT modify files yet" risk)? `<obs>`
- **AXI** — fewer redundant grep/glob/file_read re-issues? `<obs>`
- **adaptive_rounds** — how many trials took an extension, and did any of
  them convert to a pass? `<obs>`
- **sliding_window** — any trial long enough to trigger it? invariant
  violations? (expect 0) `<obs>`

## Cost / cache note

Prompt-cache gain from `stable_prompt` is still **not measurable on
Copilot** (provider not cache-eligible; no usage in streaming). Unchanged
from wave 1 — needs an Anthropic-direct run to quantify.

## Decision (fill in)

- If wave 2 > baseline with no regressions → promote flags to always-on in
  the Harbor adapter (drop the `NORN_WAVE2` gate), consider flipping
  `sliding_window` / others to on by default per the staged plan.
- If flat or mixed → keep behind flags; use the per-axis read to decide which
  axis carried its weight and which to drop or rework.
- If regressions → `<which tasks, likely cause, which flag to isolate>`.

## Caveats

- Single attempt (k=1), no variance estimate.
- `--agent-timeout-multiplier 1.5` vs the baseline re-run's 2× — note if any
  timeout reappears; not strictly A/B on the time axis.
- auto_plan adds an extra turn (input tokens ~2× the instruction) and
  adaptive_rounds can add rounds → wave 2 is expected to cost more per task;
  record runtime for the cost/benefit call.
