# Full audit of the Norn↔Terminus-2 runs + prioritised backlog (2026-09-04)

Audit of all five Norn arms and the Terminus-2 reference on the 20-task
stratified tb2 subset (`github_copilot/gpt-5.3-codex`, colima vz+Rosetta).
Goal: did we miss a harness-fixable failure, and what in the backlog is worth
next month's quota.

## Per-task matrix (Terminus-2 vs Norn best-of D/E)

| Outcome | n | tasks |
|---|---|---|
| both pass | 9 | break-filter, count-dataset-tokens, fix-code-vuln, fix-git, headless-terminal, kv-store-grpc, reshard-c4-data, sqlite-with-gcov, winning-avg-corewars |
| **T2 only** | 3 | circuit-fibsqrt, git-multibranch, qemu-startup |
| Norn only | 2 | dna-insert, query-optimize |
| both fail | 6 | model-extraction-relu-logits, overfull-hbox, protein-assembly, qemu-alpine-ssh, raman-fitting, torch-tensor-parallelism |

**Union Norn (D∪E) = 11/20 vs T2 12/20 — a one-task gap, not two.** The
per-arm 10-vs-12 overstated it.

## What the audit found that we had missed

### 1. Persistent-service tasks — the biggest real gap (3 tasks)
git-multibranch, qemu-startup, qemu-alpine-ssh. On git-multibranch Norn
**started nginx correctly** (pid 5034 on :8443, seen in its own `ss`) and
declared PASS — but the verifier runs against a **reset container where only
files persist, not processes**, so the service is gone at grading. Norn
verified in its own live session; the grader does not share it.

Terminus-2 passed by using a **git `post-receive` hook** — deployment
triggered by the grader's own push, not a process left running. That is the
competence: on "stand up a service" tasks, prefer a **triggered/persistent
mechanism** (hook, systemd unit, init script, cron) over an in-memory
process. This is reasoning we can *nudge*, plus a *verification* fix: our
`SELF_VERIFY_PROMPT` checks the deliverable in-session; it should check "as
the grader will — from a clean state, processes not assumed running".

### 2. Constraint compliance — overfull-hbox (1 task, cheapest fix)
3/4 subtests pass (compiles, no overfull hbox, protected files untouched);
fails only the check that edits stayed within the allowed synonym set. The
agent solved the task then broke an explicit stated restriction. Flagged in
the 2026-09-03 frontier analysis, still not implemented. Verify phase checks
the deliverable, never the *restrictions*.

### 3. Empty `bash` call — a small regression from the list tool (2 tasks)
circuit-fibsqrt, reshard-c4-data: the model sometimes calls `bash` with
neither `command` nor `commands`; we return a hard `InvalidArgument`, wasting
the round. Only appears in the list-tool arm (0 before). Cheap: treat an
empty bash call as a no-op nudge ("provide a command"), like the empty-LLM
retry, instead of a hard error.

### 4. Confirmed not-a-bug
- Empty-LLM-response give-up: **fixed** (`ed232aa`), verified working on
  circuit-fibsqrt (agent resumes, 2/3 subtests now).
- break-filter XSS refusal: **non-deterministic** (passed on arms D & E),
  not a fixed block.
- The remaining both-fail 4 (torch 9/13, raman 1/3, model-extraction 0/1,
  protein-assembly 0/1) are **substance on hard compute/domain tasks** —
  model frontier, not harness.

## Verdict on the harness campaign

Efficiency work paid: 416 → 238 LLM calls per 20 tasks (−43%), same score,
via three keepers (batch prompt, list tool, empty-retry). Capability tuning
(adaptive_rounds) did not. Of the true gap, **~2 tasks are harness-fixable**
(persistence reasoning + constraint verification), the rest is model.

## Prioritised backlog for next month

Ordered by (impact on this task set) × (cheapness), harness-fixable first:

1. **Constraint-verification in the verify phase** — parse the task's stated
   restrictions ("only edit X", "do not modify Y", "using only Z") and check
   them before PASS. Targets overfull-hbox directly; cheap; pure prompt +
   a check. *New, from this audit.*
2. **Persistence framing** — one prompt line: "a service/deploy must survive
   a restart; use a hook / systemd unit / init script, not a backgrounded
   process", and make SELF_VERIFY test from a clean state. Targets 3 tasks;
   cheap; the highest-count lever. *New, from this audit.*
3. **Empty-bash-call no-op** — mirror the empty-LLM retry for empty tool
   args. Trivial; recovers wasted rounds on 2 tasks. *New, from this audit.*
4. **Skill bank from trajectories (CODESKILL-lite)** — the only backlog item
   with a *measured* tb2 gain (+9.69 pts). We own the hard parts (bench guard
   = validation-gated retention/rollback; record/replay = trajectories);
   missing extraction + retrieval. Larger, but the only lever that attacks
   the substance gap by reuse rather than model swap. *Existing (wave-3 #3).*
5. **Stronger model on the same subset** — opus-5 / gpt-5.6-sol vs the 32/89
   history and vs gpt-5.3-codex here. The single biggest score lever
   (leaderboard leaders 88-92% live in this catalogue); one run when quota
   resets. *Existing.*

Deferred as before: AXI on remaining tools (bash/coderag), dynamic_tools,
sandbox Linux, Self-Harness, Viktor code-writing, micro-kernel.

## Method caveats
n=20, k=1: per-task calls are anecdotal; treat "one-task gap" and the
category counts as the signal, not any single win/loss. Norn action counts
under-counted in list mode (args not in console), which widens — not
narrows — the measured effort gap.
