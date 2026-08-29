# Terminal-Bench 2 — Harbor Run (2026-08-28)

First external, reproducible score for Norn via the Harbor adapter
(`benchmarks/harbor_adapter/norn_agent.py`).

## Setup

- Dataset: `terminal-bench/terminal-bench-2` (89 tasks, Docker-per-task).
- Harbor 0.22.0 in an isolated venv (Norn's own deps conflict with harbor's).
- Agent: Norn @ `main` (SOTA v2 wave 1), all router tiers →
  `github_copilot/claude-sonnet-4.5`; `max_tool_rounds=50`, `auto_verify`,
  `vision_tools` on (adapter defaults).
- Concurrency 4, 1 attempt/task, no timeout multiplier. Runtime: **4h57m**.

## Headline

**Reward mean: 0.360 — 32/89 tasks passed.**

That is the official Harbor number (denominator = all 89 tasks). For context,
top public tb2 agents sit around 45-55%; a bare CLI agent on sonnet is
typically in the 30s. Norn lands mid-pack on its first external run.

### Update (2026-08-29): infra artefacts eliminated, score unchanged

The first run had 18 infra exceptions (9 install failures, 9 agent
timeouts) masking whether those tasks were real capability gaps. Both
classes were fixed and the affected tasks re-run:

- **Install failures (8)** → adapter now provisions Python 3.11 via `uv`
  into a dedicated venv (commits below). All 8 install and run now.
- **Timeouts (9)** → re-run with `--agent-timeout-multiplier 2`.

Result of the re-runs: **0 tasks recovered.** Every previously-excepted task
now runs to completion and **fails the task on its merits** (reward 0) — none
were held back by infra. The timeouts in particular were not a time problem:
at 2× budget they still fail, i.e. these are compute/long-horizon tasks Norn
cannot solve, not slow ones. So **32/89 is a clean number** — every
non-pass is a genuine agent failure, no exceptions left.

Adapter fixes: `be34ca8` (dedicated venv), `04b17d2` (Python 3.11 via uv for
old-python images — qemu-alpine 3.9, mteb 3.10).

## Breakdown

| Outcome | Count | Notes |
|---|---|---|
| Passed (reward 1.0) | 32 | genuine solves |
| Failed (reward 0.0) | 49 | agent ran, verifier failed |
| — of which agent timeout | 9 | hit the wall on very long tasks |
| — of which agent crashed | 1 | `pytorch-model-recovery` (norn exit 2) |
| Install-failed | 8 | **adapter bug, not agent** — see below |

On the first run, adjusted for the (now-fixed) install bug it was 32/81 =
39.5%; after fixing install so all 89 run, it is **32/89 = 36.0%** — the
install-recovered tasks all fail on merits, so the honest denominator is 89.

## Adapter bug (fixed)

8 tasks never ran Norn on the first run — the in-container install failed for
three distinct reasons across images: pip too old for
`--break-system-packages`; Debian-owned packages that can't be uninstalled
for upgrade (urllib3 RECORD missing); and Python 3.9/3.10 images where Norn's
`>=3.11` deps can't resolve. Fixed by provisioning a standalone CPython 3.11
via `uv` into a dedicated `/opt/norn-venv` (commits `be34ca8`, `04b17d2`). All 8
now install and run; none pass the task, so the fix cleaned the measurement
rather than raising the score.

## What passed vs failed

- **Strong**: git surgery (fix-git, git-leak-recovery, git-multibranch),
  DB/SQL (sqlite-*, query-optimize), security (crack-7z-hash,
  openssl-selfsigned-cert, fix-code-vulnerability), scientific-stack
  modernization, proofs (prove-plus-comm), data merging, regex.
- **Weak**: heavy-compute / long-horizon builds (build-pov-ray,
  make-mips-interpreter, path-tracing, protein-assembly), GPU/torch
  parallelism, cryptanalysis, anything needing many sequential minutes
  (the 9 timeouts cluster here).

## Caveats

- Single attempt (k=1); no variance estimate.
- Copilot provider → no token/cost accounting from harbor (all null), and no
  prompt-cache metrics (provider not cache-eligible in Norn).
- Raw job data: `scratchpad/harbor/jobs/tb2-full/` (not committed — large;
  `result.json` summary reproduced above).
