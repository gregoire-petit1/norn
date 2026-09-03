# The 8 tasks both harnesses fail — analysis (2026-09-03)

Source: the paired harness-debt run (Norn 8/20, Terminus-2 10/20, same model
`gpt-5.3-codex`). These 8 are where *neither* harness scores, i.e. the real
frontier. They are not one problem — they decompose into four very different
kinds, and only three are genuine capability gaps.

## Finding that reframes the whole run: 17/20 tasks run x86-emulated

The task images are x86_64; our colima VM is arm64, so Docker runs them
under **QEMU emulation** (`<jemalloc>: This is the expected behaviour if you
are running under QEMU`, `downloading uv x86_64-unknown-linux-gnu`,
`ERROR: unknown platform bitness`). 17 of 20 tasks in **both** arms.

Consequences:
- Absolute scores are depressed and timeouts are manufactured. This is very
  likely why Terminus-2 scored 50% here against 64.7% published for the same
  harness+model on full tb2.
- The **A/B remains valid** — the handicap is symmetric and per-task identical.
- The Aug 28 baseline (32/89) ran on Docker Desktop, which uses **Rosetta**
  for x86 on Apple Silicon — far faster than QEMU. So today's absolute
  numbers are not comparable to that run either.

**Fix**: restart colima as `colima start --vm-type vz --vz-rosetta` (macOS 13+)
for near-native x86 speed. Expected to recover part of the timeout failures.

## Breakdown of the 8

| Task | Kind | Evidence |
|---|---|---|
| qemu-alpine-ssh | infra + long-running daemon | AgentTimeout both arms, 900 s cap, emulated |
| qemu-startup | infra + long-running daemon | AgentTimeout both, deliverable never written |
| query-optimize | **infra artefact** | verifier itself dies: `head: error reading '/proc/self/exe'`, `unknown platform bitness` — emulation breakage, not an agent failure |
| overfull-hbox | **instruction compliance** | 3/4 tests PASS (compiles, no overfull hbox, protected files untouched) — fails only `test_input_file_matches`: edits went beyond the allowed synonym substitutions |
| torch-tensor-parallelism | **near miss** | Norn 9/13 subtests vs Terminus-2 5/13 — binary reward hides real progress |
| break-filter-js-from-html | genuine: adversarial security | must craft an XSS bypass of a filter; identical selenium failure both arms |
| protein-assembly | genuine: domain knowledge | FRET/DHFR biology; Norn produced no deliverable, T2 produced a wrong one |
| raman-fitting | genuine: numerical precision | G peak fitted, 2D peak off — 1/3 subtests pass both arms |

So: **3 infra-induced, 1 compliance, 1 near-miss, 3 genuinely hard.**

## What is actually actionable for the harness

1. **Constraint compliance (overfull-hbox)** — the agent solved the problem and
   then broke an explicit rule stated in the instruction. This is the single
   clearest harness-fixable failure in the set: extract the task's stated
   constraints up front and verify against them before finishing (our
   `SELF_VERIFY_PROMPT` checks the deliverable, not the *restrictions*).
2. **Partial credit as a signal (torch)** — Norn is nearly twice as close as
   Terminus-2 on subtests yet both score 0. Tracking subtest ratios would give
   a far denser learning signal than binary reward, and would have shown Norn
   ahead here.
3. **Long-running daemons (qemu×2)** — both harnesses time out on "start a
   service and leave it running". Neither has a background-process model.
   Real, but partly an emulation artefact — re-measure with Rosetta first.

Not harness problems: break-filter (adversarial reasoning), protein-assembly
(domain knowledge), raman-fitting (numerics). Those move with the model, not
the scaffold.

## Recommended next steps

1. Re-run with `--vm-type vz --vz-rosetta` before drawing any absolute
   conclusion — 3 of the 8 are plausibly infra.
2. Add constraint extraction + verification to the verify phase (cheap, one
   prompt change, directly targets a confirmed failure).
3. Record subtest pass ratios alongside binary reward.
