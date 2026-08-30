# Norn Wave 3 — Instrumentation & Harness-Debt Experiment (2026-08-31)

## Context

tb2 baseline: **32/89 (36%)**, and it is a *clean* number — the 18 infra
exceptions were fixed and recovered **0 tasks** (`2026-08-28-terminal-bench-2-harbor.md`).
The ceiling is harness/capability, not plumbing.

Market data says that is where the leverage is:

| Finding | Source |
|---|---|
| Same model, different scaffold: **13.7–20.8 pts** swing | Scaffold Effect (arXiv 2607.22585) |
| Claude Opus: 93% (Cursor) vs 77% (Claude Code) | Harness Effect |
| **Terminus 2** (reference harness) 64.7% w/ GPT-5.3-Codex; SageAgent 78.4% same model | Scaffold Effect |
| Harness moves **tokens-per-solved-task ~40×**; model only 1.0–1.3× | Scaffold Effect |
| Goose ~28K vs OpenCode ~1.1M tokens/solved — *same* pass rate (48–50%) | Scaffold Effect |
| Skills learned from trajectories: **+9.69 pts** (tested on tb2); SAGE −26% steps, −59% tokens | CODESKILL, SAGE |

## #1 — Instrumentation (DONE, commit `69931e2`)

We could not compute tokens-per-solved-task: Copilot reports no usage, every
token field was 0. Now:

- `_local_token_usage()` — litellm tokenizer fallback (messages + tool schemas
  + response). Verified on our Copilot model id.
- `llm.complete()` / `llm.stream()` use it when the provider reports nothing;
  every event carries **`token_source`** = `provider` | `local` | `unavailable`.
  Opt-out `count_tokens_locally=False` restores the old zeroes contract.
- Bench: `TaskResult.prompt_tokens/completion_tokens/token_source`; the report
  prints per-task tokens and the headline **tokens per solved task**.

**Caveat**: litellm falls back to a generic tokenizer for these model ids, so
counts are comparable *across our own runs* (what A/B needs) — not billing truth.

## #2 — Harness debt (READY, blocked on quota until 2026-09-01)

We do not know how much of our 36% is the model (sonnet-4.5, not a 2026
frontier model) versus our harness. One run settles it: **same model, same 89
tasks, reference harness**.

Pre-flight (quota must be > 0):
```
curl -s -H "Authorization: token $(cat ~/.config/litellm/github_copilot/access-token)" \
  -H "Editor-Version: vscode/1.90" https://api.github.com/copilot_internal/user \
  | python3 -c "import json,sys; print(json.load(sys.stdin)['quota_snapshots']['premium_interactions'])"
```

Run (validated with `--print-config`; harbor drives litellm from the host, so
it reuses the same cached Copilot OAuth token as Norn):
```
<harbor-venv>/bin/harbor run \
  -d terminal-bench/terminal-bench-2 \
  -a terminus-2 -m github_copilot/claude-sonnet-4.5 \
  -o <jobs-dir> --job-name tb2-terminus2-baseline -n 4 -q -y
```

Reading the result:

| Terminus-2 score | Interpretation | Next move |
|---|---|---|
| ~55–60% | **~20 pts of pure harness debt** | Attack context discipline + stop/recovery policy (#4, #5) |
| ~45–55% | Moderate debt | Same, plus model routing worth testing |
| ~35–40% | Our model is the ceiling | Route to a frontier model first; harness work has less headroom |

Caveats: tb2 vs tb2.1 versions differ across published numbers; token
comparison Norn-vs-Terminus2 is **not** apples-to-apples (our local counter
vs harbor's provider-reported path, which Copilot leaves empty) — the
**pass rate is the valid signal**, token comparison needs one counting method.

## Backlog (ordered, post-#2)

3. **Skill bank from trajectories** (CODESKILL-lite) — the only lever measured
   *on tb2* (+9.69 pts). We already own the two hard parts: `norn bench guard`
   (validation-gated retention + rollback) and record/replay (trajectories).
   Missing: extraction + retrieval.
4. **Context discipline** — pre-injection vs accumulation; where the 40× lives.
5. **Failure-signature policy** — REASON / VERIFY / TIME are distinct harness
   signatures; make ours explicit and pick a stop policy.
6. **Self-Harness** — prompts/decomposition self-edit, retained only if the
   bench validates (the guard is already the required safety net).
