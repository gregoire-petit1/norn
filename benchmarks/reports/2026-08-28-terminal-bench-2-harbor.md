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

## Breakdown

| Outcome | Count | Notes |
|---|---|---|
| Passed (reward 1.0) | 32 | genuine solves |
| Failed (reward 0.0) | 49 | agent ran, verifier failed |
| — of which agent timeout | 9 | hit the wall on very long tasks |
| — of which agent crashed | 1 | `pytorch-model-recovery` (norn exit 2) |
| Install-failed | 8 | **adapter bug, not agent** — see below |

**Adjusted for the adapter bug: 32/81 = 39.5%** on tasks where Norn actually
installed and ran.

## Adapter bug (actionable)

8 tasks never ran Norn at all — the in-container `pip install -r
/opt/requirements.txt` failed (`build-pmars`, `install-windows-3-11`,
`mailman`, `mteb-leaderboard`, `mteb-retrieve`, `qemu-alpine-ssh`,
`qemu-startup`, `winning-avg-corewars`). The `uv export`-generated
requirements file is rejected by these task images' pip (usage error /
non-zero exit). These count as reward 0 in the official number but reflect an
adapter packaging issue, not agent capability. Fix candidates: pin
`pip`/`setuptools` before install, or vendor a wheelhouse instead of a
resolve-at-install requirements file.

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
