# Batching experiment — four arms, same 20 tb2 tasks, same model (2026-09-03/04)

Model `github_copilot/gpt-5.3-codex`, colima vz+Rosetta, harbor 0.22.0,
sequential arms. Reference: Terminus-2 12/20 (60%), 167 LLM calls, 21 min.

| Arm | Change | Pass | LLM calls | Tool calls | Prompt tokens (local est.) | Wall |
|---|---|---|---|---|---|---|
| B | baseline (old prompt) | 8/20 | 416 | 474 | — | 57 min |
| C | prompt: batching rule | 9/20 | 261 | 230 | 2.10 M | 45 min |
| **D** | **C + `bash(commands=[...])`** | **10/20** | **238** | **205** | **1.76 M** | **42 min** |
| T2 | reference harness | 12/20 | 167 | 566 | n/a | 21 min |

## D: the list tool is used, and it is the mechanism that works

- **87% of bash calls use the list** (137/157); batch size averages ~3.9
  commands (sampled from failing batches; max 25).
- LLM calls down another 9% vs C (261 → 238) and **43% vs B** (416 → 238).
  Prompt tokens down 16% vs C.
- Pass rate 10/20 — highest Norn arm, equal to Terminus-2 under QEMU; still
  below T2's 12/20 under Rosetta. Newly passing vs C: break-filter-js-from-html,
  query-optimize. Lost: git-multibranch. All within n=20 noise.
- The tool behaved as designed in production: `$ cmd [rc=127]` sections, stop
  at first failure, remaining commands reported as not run; the model
  recovered on the next turn.

## Reading across the arms

1. **Wording removal (B → C)** gave the biggest single drop in LLM calls
   (-37%) — but by making the model do *less* (tool calls halved), not by
   batching. Same score, so the removed exploration was not buying anything.
2. **Structural batching (C → D)** gave a further, genuine efficiency gain:
   tool *calls* fell slightly while the *commands* they carry rose (~3.9 per
   list call), i.e. more work per round-trip. This is the Terminus-2
   mechanism, now available to Norn with native function-calling (no
   text-parser artefacts like T2's `rm -f "tash list"`).
3. Norn is now at **~12 LLM calls/task** vs Terminus-2's 8.4, from 21. The
   remaining gap is mostly that T2 does ~2.5× more tool work per task (28 vs
   ~11 commands) *and* still uses fewer round-trips — it explores more,
   cheaper. Whether Norn's lower exploration costs it the last 2 tasks is the
   next question.

## Cost

Whole 4-arm campaign (B, C, D + one T2 replay): ~1,100 premium interactions.
D alone ≈ 240 — the cheapest Norn arm, as expected since calls fell.

## Next

- Keep both changes (prompt `49542b7`, list tool `e49af8a`); `batch_size` on
  `tool.call` events (`29e0592`) lands in the next run so batching is measured
  from events, not console parsing.
- Add the `prompt_tokens` sum to the per-arm table as a first-class metric
  (now possible from JSONL) — tokens-per-solved-task is the number to drive.
- Investigate the exploration gap: does Norn under-inspect before acting?
  Compare read-only-command counts per task against T2.
