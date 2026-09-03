# Harness Debt Experiment — Norn vs Terminus-2, same model (2026-09-03)

**Question**: how much of Norn's tb2 ceiling is the harness versus the model?
**Method**: paired A/B — both harnesses, same 20 tb2 tasks, same model
(`github_copilot/gpt-5.3-codex`), same runner (harbor 0.22.0), same day.

## Headline

| Arm | Score | Runtime |
|---|---|---|
| Terminus-2 (reference harness) | **10/20 = 50%** | 1h07 |
| **Norn** (wave-1 + wave-2 code, wave-2 flags OFF) | **8/20 = 40%** | 1h03 |

**Gap: 10 points (2 tasks) — NOT statistically significant.** McNemar exact
on the 6 discordant pairs (4 vs 2): **p = 0.69**.

Paired breakdown: both pass 6 · Terminus-2 only 4 (circuit-fibsqrt,
dna-insert, git-multibranch, reshard-c4-data) · Norn only 2
(model-extraction-relu-logits, winning-avg-corewars) · both fail 8.

## What this changes

1. **Norn is competitive with the reference harness.** We expected a ~20-point
   debt extrapolating from the old 36%. There is no measurable debt at n=20.
2. **The old 36% was mostly the model.** It was measured on
   claude-sonnet-4.5, which Copilot retired in Sept 2026. On a current model
   Norn reaches 40% on a subset where it historically scored 35%, and
   Terminus-2 — a harness published at 64.7% on full tb2 with this exact
   model — lands at 50% here, i.e. this subset is harder than tb2 average.
3. **Priority shifts.** Harness work is no longer the obvious first lever;
   closing the remaining gap means beating a harness we are already level
   with. The 8 tasks *both* harnesses fail are the real frontier.

## Production bug found by the run (fixed)

Norn sends `temperature=0.0` on every call. The gpt-5 family accepts only
`temperature=1` and rejects the request: **13/13 trials died in ~164 ms**
with `UnsupportedParamsError` before doing any work — a 0/13 that looked like
a capability result. Fixed by `litellm.drop_params = True` (commit
`8a6d619`). Without it Norn cannot use **any** gpt-5 model, i.e. most of the
2026 frontier lineup. No unit test could have caught this; it took a real
provider.

## Caveats

- n=20, single attempt: the 10-point gap is within noise (p=0.69). Treat
  "roughly level" as the finding, not "Norn is 10 points behind".
- Subset is stratified on *historical* (sonnet-4.5) outcomes, so it is
  representative of tb2 difficulty as Norn saw it then, not of tb2 overall —
  consistent with Terminus-2 scoring 50% here vs 64.7% published.
- 4 tasks hit harbor's 900 s cap in each arm (mostly the same ones:
  qemu-*, query-optimize) and count as failures in both.
- Token comparison between arms is not valid (Norn counts locally, harbor
  uses the provider path which Copilot leaves empty).

## Cost

1595 / 3900 premium interactions for the month, including ~1000 wasted on two
operational errors (a run left going in parallel; the broken-mount arm).
~28 interactions per task with gpt-5.3-codex.
