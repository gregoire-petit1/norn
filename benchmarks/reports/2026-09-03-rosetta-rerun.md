# Rosetta re-run — Norn vs Terminus-2, same model, no QEMU (2026-09-03)

Same paired design as the harness-debt experiment (20 tb2 tasks,
`github_copilot/gpt-5.3-codex`, harbor 0.22.0), but the colima VM was
rebuilt with `--vm-type vz --vz-rosetta`: x86 task images now run under
Rosetta instead of QEMU (verified: `binfmt_misc/rosetta` enabled, a 3M-iter
Python loop at 0.13 s ≈ native, no `running under QEMU` in any container).

## Headline

| Arm | QEMU (03/09 am) | **Rosetta** | Runtime |
|---|---|---|---|
| Terminus-2 | 10/20 | **12/20 = 60%** | **21 min** |
| Norn | 8/20 | **8/20 = 40%** | **57 min** |

Terminus-2's 12th pass is `fix-code-vulnerability`, replayed alone after a
`buildx`-missing compose failure during the batch (infra artefact; it passed
under QEMU for both arms and passes here in 2 min). buildx is now installed.

**Gap: 20 points (4 tasks). McNemar exact p = 0.38 — still not significant
at n=20**, but the gap doubled vs QEMU and every discordant task but one
favours Terminus-2 (T2 only: break-filter-js-from-html, circuit-fibsqrt,
git-multibranch, qemu-startup · Norn only: none after the replay).

Terminus-2 at 60% is now within reach of its published 64.7% — the run
environment is finally credible. Norn did not move.

## The real finding: Norn spends ~2.3× the LLM calls of Terminus-2

| | Terminus-2 | Norn |
|---|---|---|
| Median wall-clock per task | 153 s | **300 s** |
| Median LLM calls per task | 9 steps | **21 calls** |
| Tool execution time (fix-git) | — | 2 s of 165 s |

Norn is slower on **19 of 20 tasks**, including the ones it passes — this is
not "trying harder on hard tasks", it is a per-task overhead. Tool time is
negligible; the wall-clock is LLM round-trips. And it is **not auto_verify**:
on protein-assembly, 38 of 40 calls are in the main loop, 2 in verification.

Consequences:
- ~2× the premium-interaction cost per task for the same or lower pass rate.
- Directly the metric the Scaffold Effect paper says the harness controls
  (tokens/steps per solved task). Terminus-2 reaches the same result in
  fewer, larger steps; Norn takes many small ones.
- This is the most concrete, harness-owned inefficiency found so far —
  cheaper to fix than any capability gap, and it compounds with everything.

Hypotheses to test (ordered): one tool call per step vs Terminus-2 batching
several shell commands per turn; exploration-heavy prompting (Norn issues
5+ read-only `git`/`ls` calls before acting on fix-git); the 50-round cap
inviting long tails. Needs the per-call JSONL from inside the container,
which the adapter does not currently ship out — add `~/.norn/logs` to the
agent artifacts.

## Noise found on the way

Every LLM call logs `PydanticSerializationUnexpectedValue: Expected
ResponseAPIUsage`. litellm routes gpt-5.3-codex via the Responses API and our
usage parsing expects the chat-completions shape. Harmless to results
(token accounting falls back to the local counter) but it is 4 lines of
stderr per call and a sign the Responses path is not first-class yet.

## Revised reading of "the 8 both fail"

Under Rosetta only **7** remain failed by both, and the composition changed:
`break-filter-js-from-html` and `qemu-startup` — which I had filed as
"genuinely hard" and "infra" — are now passed by Terminus-2. Two of the
"genuinely hard" three were partly emulation after all. Remaining common
failures: dna-insert, model-extraction-relu-logits, overfull-hbox,
protein-assembly, qemu-alpine-ssh, query-optimize, raman-fitting,
torch-tensor-parallelism (Norn 9/13 subtests here — still a near miss).

## Caveats

n=20, k=1; gap not significant. Runtime comparison is fair (same VM, same
day, sequential arms, 4 concurrent trials each). LLM-call counts for Norn
are inferred from a per-call warning line, not from event logs.
