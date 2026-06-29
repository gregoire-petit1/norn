# Norn SOTA Plan — 2026-06-20

Source: notes from AI engineer colleague (harness track) + AI-generated draft plan (reworked) + analysis of terminal-bench (harbor-framework) and KIRA (krafton-ai/Terminus-KIRA) repos.

---

## Context

Best benchmark: 14/16 (88%) — Wave 1 token opt  
Latest benchmark: 12/16 (75%) — Ollama 429 rate limits (free-tier artifact)  
Current HEAD: `90cc321` feat(repo-map): wire into AgentLoop

Already done: AST repo map, token opt Wave 1+2, env bootstrap, sliding window context, Phase 10 harness.

**Market framing:** 429s are a dev-environment artifact of free-tier providers (Ollama cloud, OpenRouter free). On paid APIs (Claude API, OpenAI, Mistral, etc.) rate limits are not the bottleneck — **cost per task** is. At scale, 80% prompt cache hit rate = 80% cost reduction. That's the competitive moat. Viktor cache invariants are therefore the correct Phase 1 for production relevance, not just an architectural nicety.

**Revision vs AI-generated plan:** The AI plan prioritized Viktor correctly for production. Initial re-ordering was correct only for current free-tier dev setup. Plan below targets paid-provider production use.

---

## Re-assessment of the 5 tracks

### Track 1 — Viktor AI (SDK-first tools)
The colleague calls this "le plus sous-estimé". Correct, but the full Viktor pattern is two separate ideas:

**Idea A** — Byte-stable prefix + append-only thread + in-cache compaction  
→ **Separable, low-risk, high ROI.** Can be done without paradigm shift.

**Idea B** — Replace tool schemas with "agent writes Python code"  
→ **Major paradigm shift.** Requires Python sandbox, re-prompting strategy, and likely breaks existing benchmarks during transition. High ceiling, high risk.

These should be sequenced separately. A before B.

### Track 2 — RAMP (Map + Proof)
/plan and /proof slash commands directly improve benchmark score on hard tasks:
- Plan mode → better task decomposition before coding
- Proof mode → self-verification (run tests, check outputs) before returning

Already have Rules (AGENTS.md/memory) and partial Augment (tools/mcp). Map + Proof = the missing half.

### Track 3 — Self-improvement
AGENTS.md auto-update is the most pragmatic version. Pattern:  
post-task reflection → extract failure lessons → persist → improve next run.

Norn already has post-task hook slots from harness engineering (Phase 10). Wire them up.

### Track 4 — Sandbox / filesystem
Least benchmark-relevant right now. Deferred to last.

### Track 5 — CodeRAG++
Builds on existing `repo_map.py` (AST-based). The jump to embeddings + hybrid search is meaningful but complex. Medium priority.

### Track 6 — terminal-bench + KIRA (new source)

**terminal-bench** (harbor-framework) = external benchmark harness. ~100 real terminal tasks, Docker-sandboxed, oracle solutions + verification scripts, `tb` CLI, public leaderboard.  
**KIRA** (krafton-ai/Terminus-KIRA) = agent tuned to score on terminal-bench. Tricks: marker-based command-completion polling, output caps, multimodal, multi-perspective verification, Anthropic prompt caching.

Framing: these are complementary. terminal-bench = the test. KIRA = an agent that games the test well.  
**Norn overlaps KIRA** (both are agents) and **could consume terminal-bench** (the harness).

What Norn already has — skip:
- Native function calling — Norn default; `tool_call_extractor.py` is fallback only
- Output truncation — `max_tool_result_chars=8000` already in loop
- Context summarization — `context.py` + sliding window
- Verify phase — coordinator has research/build/verify

**Net-new from this track:**

**A — terminal-bench adapter** (highest value from this track)  
Norn's internal benchmark suite is non-comparable externally. Write a `tb` adapter so Norn runs under terminal-bench → industry-standard numbers + leaderboard position. Doesn't replace internal suite; adds external yardstick.  
Caution: terminal-bench is v0.1.1/beta, dataset schema may churn → pin adapter to a dataset version.

**B — Marker-based command polling** (practical win)  
`bash_tool.py` = blocking `asyncio.wait_for(process.communicate(), timeout)` — fixed timeout, kills on overrun, no long-running support.  
KIRA approach: echo `__CMDEND__<seq>__` marker to detect completion early instead of fixed wait → handles long builds/trains/bg processes.  
Impact: unblocks "long-horizon" task category (already in Norn's benchmark suite).

**C — Prompt caching via litellm** (maps to Phase 1)  
KIRA uses Anthropic `cache_control`. Norn is model-agnostic via litellm — litellm exposes `cache_control` for Anthropic behind provider check. Wire in `llm.py`. Same track as Viktor byte-stable prefix.

**D — Multi-perspective verification checklist** (maps to Phase 2)  
Pure prompt change: validate output as test-engineer + QA + user. Drop into coordinator verify phase / /proof command. Near-zero cost.

**E — Docker sandbox per task** (maps to Phase 6, scoped)  
Phase 11 design has `sandbox.py` (tmpdir), not Docker. terminal-bench uses Docker per task → reproducibility + no host contamination. Upgrade path for `benchmarks/runner/` specifically (narrower scope than full YoloFS).

Skip from KIRA: `image_read` multimodal (not core to Norn's MLOps angle), Terminus 2 runtime.

---

## Priority order (by Norn-specific ROI)

### P0 — Dev-env patch: 429 retry + benchmark re-run (1 day, low priority)

**Scoping:** This is a free-tier workaround, not a product priority. On paid providers it disappears. Worth a minimal fix so benchmarks are meaningful, nothing more.

Tasks:
- Exponential backoff with jitter on 429 in `llm.py` / `format_llm_error()`
- `max_retries` config flag (default: 3)
- Benchmark re-run to confirm true baseline with Wave 2 + repo map

**Not a gate.** Viktor work (Phase 1) can start in parallel — the two are independent.

---

### Phase 1 — Viktor Quick Wins: stable cache (1 week)

Goal: implement the three Viktor invariants that don't require paradigm shift.

**W1.1 — Byte-stable system prompt**  
Tool schemas currently injected into system prompt → any tool change invalidates cache.  
Fix: move tool schemas OUT of system prompt header. System prompt = static. Schemas pass via separate API field or first user turn.  
Expected: 50-80% cache hit rate improvement.

**W1.2 — Append-only thread invariant**  
Enforce: nothing in the conversation thread is ever modified. Summarization must append a `<summary>` turn, not replace prior turns.  
Check: `sliding_window.py` / context manager — verify no in-place edits.  
Current compaction (summary_provider) may already do this; confirm and add assertion.

**W1.3 — In-cache compaction**  
Summarization request = append to active thread, not a separate API call with different prefix.  
Same API call, same prefix → cache hits carry over.  
Impact: compaction cost drops ~4x.

**W1.4 — Tool schema minification (already partially done)**  
Verify schema minify from Wave 1 is still active after repo map / env bootstrap additions.

**W1.5 — Prompt caching via litellm `cache_control`**  
litellm exposes `cache_control` for Anthropic provider. Wire into `llm.py` behind `if provider == "anthropic"` guard.  
Synergy: byte-stable prefix (W1.1) is prerequisite — cache_control only helps if the prefix is stable.

---

### Phase 1.5 — Harness: terminal-bench adapter + marker polling (1 week, parallel-able with Phase 2)

**W1.5a — terminal-bench adapter**  
Write `benchmarks/adapters/terminal_bench.py`: implement the `tb` agent adapter interface so Norn runs under `tb run`.  
Pin to terminal-bench dataset version (e.g. `v0.1.1`) in adapter config — schema churn is real at beta.  
Result: Norn gets external reproducible numbers + leaderboard position alongside internal suite.

**W1.5b — Marker-based command polling in `bash_tool.py`**  
Replace `asyncio.wait_for(process.communicate(), timeout)` with streaming read loop that polls for `__CMDEND__<seq>__` marker.  
Fallback: if marker never arrives within timeout, kill as today.  
Benefit: long builds/installs/training runs complete naturally instead of timing out → long-horizon task category.

---

### Phase 2 — RAMP: Plan + Proof modes (1-2 weeks)

**W2.1 — /plan command**  
Agent receives task → reasons aloud → produces structured plan (numbered steps, files to touch, test strategy) → awaits confirmation → executes.  
Prompt: CoT "think before code" wrapper. Model: same qwen3-coder.  
Integration: new slash command + `AgentLoop` pre-task hook.

**W2.2 — /proof command (or auto-proof flag)**  
After task complete, agent:
1. Runs test suite (or targeted tests)
2. Checks outputs match spec
3. If fail: re-enters loop with failure context (max 2 retries)
4. Reports: PASS/FAIL + evidence

Direct benchmark impact: catches "done but wrong" completions.

**W2.2b — Multi-perspective verification prompt (from KIRA)**  
In the /proof verify prompt: explicitly instruct the agent to evaluate output from 3 lenses:
1. Test-engineer: are edge cases handled?
2. QA: does output match spec literally?
3. End-user: would this actually work in practice?

Pure prompt addition, zero code change. Orthogonal to W2.2 — add to same verify phase.

**W2.3 — /rules inspect**  
View + edit AGENTS.md and memory context inline. Lower priority, but completes the RAMP surface.

---

### Phase 3 — Self-Improvement Loop (2-3 weeks)

**W3.1 — Post-task reflection skill**  
After each task (especially failures):
- Agent answers: "What went wrong? What would I do differently?"
- Extract structured lesson: `{trigger, mistake, correction}`
- Append to AGENTS.md or dedicated `LESSONS.md`

**W3.2 — Failure pattern detection**  
Scan AgentLoop traces for:
- Infinite tool-call loops
- Give-ups (max_rounds exceeded with no result)
- Recurring error strings
Aggregate into failure report. Runs after bench suite.

**W3.3 — Benchmark regression guard**  
Before any AGENTS.md auto-update is persisted, run a mini benchmark (5 easy tasks). If regression → revert. Prevents self-improvement from self-harming.

Inspired by SICA (simplest version) + recursive-improve (trace capture) + AGENTS.md pattern.

---

### Phase 4 — CodeRAG++ (2-3 weeks)

Builds on existing `repo_map.py` (AST). Not a replacement — an upgrade path.

**W4.1 — Embedding index**  
Chunk by function/class. Embed with local model (nomic-embed-text via Ollama, or sentence-transformers).  
Store in ChromaDB (local, no infra).

**W4.2 — Hybrid search**  
BM25 (keyword) + vector (semantic) → RRF fusion rank.  
Outperforms either alone on code retrieval (per CodeRAG paper).

**W4.3 — Call-graph expansion**  
After retrieval, expand context by one hop: callers + callees of retrieved functions.  
Dependency graph from existing AST traversal.

**W4.4 — Token budget assembly**  
Given N relevant chunks, assemble context within configured token budget (e.g. 8k).  
Prioritize: direct match > one-hop caller > two-hop.

**W4.5 — MCP tools**  
Expose as: `coderag_search(query)`, `coderag_context(symbol)`, `coderag_explain(file)`.  
Integrates with existing `src/norn/mcp/` structure.

Ref: CodeRAG paper is most mature, has MCP integration design already.

---

### Phase 5 — Full Viktor Architecture: code-writing paradigm (3-4 weeks, high risk)

Only if Phases 1-4 don't push benchmark to target ceiling, OR if token costs become critical at scale.

**Paradigm:** Remove all tool schemas from API. Give agent 4 stable tools only:
- `shell(cmd)` — run any shell command
- `read_file(path)` → string
- `write_file(path, content)`
- `send_message(channel, text)` (optional)

Agent writes Python code blocks that `import stripe`, `import linear`, etc. Code executed in sandbox. Result fed back as next turn.

**Why defer:**
- Breaks existing tool-call loop architecture
- Requires Python sandbox (Docker or restricted exec)
- Benchmark regression during transition likely
- Reward: unlimited tool scaling + ~80% cache cost reduction + native composition (loops/conditionals in one turn)

**Tasks when prioritized:**
- W5.1: SandboxProvider (Docker or modal.com or restricted subprocess)
- W5.2: SdkLoader — pre-import SDK modules into sandbox namespace
- W5.3: CodeAgent mode — prompting strategy to write Python instead of tool calls
- W5.4: Migration path — hybrid (tool calls fall back to code if no schema found)

---

### Phase 6 — Sandbox / YoloFS (deferred)

Lowest benchmark ROI. Relevant for production safety, not accuracy.

**Phase 6a — Docker sandbox for benchmarks (scoped, earlier than full YoloFS)**  
Phase 11 design has `sandbox.py` (tmpdir isolation). Upgrade `benchmarks/runner/` to Docker per task:
- Reproducibility: each task starts from clean image
- No host contamination from benchmark runs
- Aligns with terminal-bench approach (comparability argument)
Scope: benchmark runner only, not agent execution sandbox. Narrower than full YoloFS.

**Phase 6b — Full YoloFS / production sandbox**
- Staging layer (all writes buffered, diff visible before commit)
- Snapshot/rollback
- Progressive permissions (read → write → write-sensitive)

Revisit after Phase 3-4.

---

## Summary table

| Phase | Track | Effort | Benchmark ROI | Risk |
|-------|-------|--------|---------------|------|
| P0 | 429 retry patch | S | Dev-env fix | Low |
| 1 | Viktor cache invariants + litellm caching | M | High (cost) | Low |
| 1.5 | terminal-bench adapter + marker polling | M | External comparability + long-horizon tasks | Low |
| 2 | RAMP Plan+Proof + multi-perspective verify | M | High (accuracy) | Low |
| 3 | Self-improvement loop | L | Medium | Medium |
| 4 | CodeRAG++ | L | Medium | Medium |
| 5 | Full Viktor paradigm (code-writing) | XL | High ceiling | High |
| 6a | Docker benchmark sandbox | M | Reproducibility | Low |
| 6b | Full YoloFS | XL | Production safety | Medium |

---

## Key divergences from AI-generated plan

**Viktor priority: AI plan was right, for the right reasons.**  
Viktor cache invariants (Phase 1 here) should be first. The AI plan was correct. Initial re-ordering assumed "free-tier stability > cost efficiency" — that's wrong for production targets.

**Full Viktor paradigm (Python code-writing): still Phase 5.**  
The AI plan conflated Viktor Idea A (cache invariants) with Idea B (code-writing paradigm). These must be sequenced. Idea A = additive, low risk. Idea B = architectural rewrite, high risk. Idea A unlocks most of the cache benefit without the risk of B.

**RAMP Plan+Proof: Phase 2, not Phase 1.**  
High accuracy ROI but orthogonal to cost. Do cost optimization first (defensible economics), then accuracy improvements (defensible quality).

**429 fix: minimal patch, not a gate.**  
Free-tier artifact. Don't block on it.
