# Norn — Token Optimization Design: Prompt Efficiency & Context Management

**Date:** 2026-04-22
**Status:** Research complete, ready for implementation planning
**Prerequisite:** Phase 9 v2 (prompt caching — workstream G) ✅
**Related phases:** Phase 10 (obsidian_rag — adds context), Phase 11 (benchmark
— measures token metrics)

---

## 1. Problem Statement

Norn currently sends ~1,958 prompt tokens for a trivial one-shot "hello"
request. In multi-turn `norn chat` sessions with tool use, this grows
rapidly due to:

- **Tool schemas**: 20+ tool definitions sent every turn (~200–500 tokens each)
- **Conversation history**: Full verbatim history including tool_call/result
  pairs accumulates linearly
- **System prompt**: Static instructions repeated every turn
- **Tool outputs**: File reads, bash results, grep output can each be 5k+
  tokens

With cloud models (especially free-tier with aggressive rate limits), this
is a critical bottleneck for both cost and reliability.

---

## 2. Goal

Reduce per-turn prompt token consumption by **40–70%** on typical multi-turn
sessions without degrading response quality. Provide a layered set of
optimizations that can be enabled incrementally.

---

## 3. Current State (What Norn Already Has)

| Technique | Status | Location |
|-----------|--------|----------|
| Anthropic/OpenAI prompt caching | ✅ Phase 9 v2 G | `src/norn/core/llm.py` |
| Basic tool output truncation | ✅ Partial | Runner level |
| Structured prompt ordering (static first) | ✅ Implicit | `agent.py` message construction |

---

## 4. Optimization Catalog

### Tier 1 — Quick Wins (low effort, high impact)

#### 4.1 Tool Schema Minification

**What:** Strip verbose descriptions, examples, and redundant type
annotations from tool JSON schemas. Keep only `name`, 1-sentence
`description`, and minimal `parameters`.

**Savings:** 30–60% on tool definition tokens.

**Complexity:** Low — single pass over `ToolDefinition.schema()` output.

**Implementation:**
- Add `minify: bool = True` option to `ToolRegistry.get_schemas()`
- Strip `description` fields longer than 80 chars → truncate or rewrite
- Remove `examples`, `default` annotations where redundant
- Remove `additionalProperties` boilerplate (inferred by most models)
- Benchmark: count tokens before/after with `tiktoken`

**Risks:** Some models rely on detailed parameter descriptions for correct
tool use. Needs A/B testing per model family (Qwen, Claude, GPT).

**Files touched:** `src/norn/tools/registry.py`, individual tool schemas.

#### 4.2 Tool Output Truncation & Summarization

**What:** Cap tool result tokens before storing in message history. Large
outputs (file reads, bash, grep) get truncated with head/tail bookends or
LLM-summarized if above threshold.

**Savings:** 30–60% on tool-heavy sessions.

**Complexity:** Low.

**Implementation:**
- Add `max_tool_result_tokens: int = 2000` to config
- Truncation strategy: keep first 60% + last 20% + `\n...[truncated N tokens]...\n`
- Optional: for very large outputs (>4k tokens), summarize with a fast
  model before inserting into history
- Apply in `agent.py` before appending tool result to `messages`

**Files touched:** `src/norn/core/agent.py`, `configs/default.yaml`.

#### 4.3 System Prompt Compression

**What:** Rewrite Norn's system prompt as dense bullet points. Remove
redundant phrasing. Current system prompt likely has 30–50% slack.

**Savings:** 20–50% on system prompt tokens.

**Complexity:** Low — but requires quality regression testing.

**Implementation:**
- Audit current system prompt token count
- Rewrite for density (bullets, abbreviations, no repetition)
- Validate with benchmark suite (Phase 11) that behavior doesn't degrade

**Files touched:** `src/norn/core/prompts.py` (or wherever system prompt lives).

---

### Tier 2 — Medium Effort, High Impact

#### 4.4 Sliding Window + History Summarization

**What:** Keep only the last N turns verbatim. Older turns get collapsed
into a condensed "conversation summary" message inserted after the system
prompt.

**Savings:** 50–80% on long sessions (>10 turns).

**Complexity:** Medium.

**Implementation:**
- Add config: `context.max_history_tokens: 8000`, `context.summary_model: null`
  (uses primary model if null)
- Token counting: use `litellm.token_counter()` or `tiktoken`
- Trigger: when total message tokens exceed `max_history_tokens`
- Summary prompt: "Summarize the conversation so far in ≤200 tokens,
  preserving key decisions, file paths mentioned, and task context."
- Architecture:
  ```
  [system_prompt] [summary_of_old_turns] [recent_N_turns] [tools]
  ```
- Summary is regenerated when window slides (not every turn — cache it)
- Pin mechanism: allow marking specific messages as "never evict" (e.g.,
  initial task description)

**Files touched:** `src/norn/core/agent.py`, `src/norn/core/context.py` (new),
`configs/default.yaml`.

**References:**
- Aider: uses "chat summaries" for this exact pattern
- Claude Code: configurable context window with smart eviction

#### 4.5 Dynamic Tool Selection

**What:** Instead of sending all 20+ tool schemas every turn, select only
the 3–8 most relevant tools based on the current conversation context.

**Savings:** 30–80% on tool schema tokens (proportional to tool count
reduction).

**Complexity:** Medium.

**Implementation options (ranked by complexity):**

1. **Keyword heuristic** (simplest): scan last user message + recent
   assistant messages for keywords → map to tool groups
   - "file", "read", "edit" → file tools
   - "run", "bash", "execute" → bash tool
   - "search", "find", "grep" → search tools
   - "model", "tensor", "dataset" → ML tools
   - Always include: bash, file_read, file_edit (core set)

2. **Embedding similarity**: embed tool descriptions + current query,
   select top-K by cosine similarity. Requires an embedder (already
   planned for Phase 10 obsidian_rag).

3. **LLM-based selection**: ask a fast model "which tools are relevant?"
   — overhead may negate savings.

4. **OpenAI Tool Search** (provider-specific): OpenAI offers server-side
   tool selection for large tool sets. Only works with OpenAI models.

**Recommendation:** Start with option 1 (keyword heuristic), upgrade to
option 2 once Phase 10 embedder exists.

**Config:**
```yaml
context:
  dynamic_tools: true          # enable/disable
  always_include_tools:        # core tools always sent
    - bash
    - file_read
    - file_edit
    - file_write
  max_tools_per_turn: 8
```

**Files touched:** `src/norn/tools/registry.py`, `src/norn/core/agent.py`,
`configs/default.yaml`.

#### 4.6 Intermediate Tool Call Collapse

**What:** After a tool_call + tool_result pair is no longer in the "recent
window" (see 4.4), collapse it into a brief summary instead of keeping
the full request/response.

**Savings:** 40–60% on multi-step agent sessions.

**Complexity:** Medium.

**Implementation:**
- When sliding window evicts a turn containing tool calls:
  ```
  # Before (verbatim in history):
  assistant: tool_call(bash, {"command": "find . -name '*.py' | head -50"})
  tool: [50 lines of output]

  # After (collapsed):
  assistant: [Used bash to list Python files — found 50 results in src/]
  ```
- Collapse happens at summarization time (part of 4.4)
- Must preserve tool names and key outcomes for coherent conversation

**Files touched:** integrated into `src/norn/core/context.py` (same as 4.4).

---

### Tier 3 — Advanced (higher effort, specialized gains)

#### 4.7 Repo Map (Tree-Sitter AST-Based Context)

**What:** Generate a condensed map of the codebase — file tree with
function/class signatures and line numbers — instead of sending full
files. The agent sees the "shape" of the code and requests specific
function bodies on demand.

**Savings:** 70–90% vs full file content. A 2000-token module becomes a
~200-token signature block.

**Complexity:** Medium.

**Prior art:**
- **Aider** (Paul Gauthier, 43.7k stars): pioneered the "repo-map"
  concept. Uses tree-sitter to parse all project files, extracts
  function/class definitions, and injects a condensed map into context.
  This is Aider's single most important context optimization. See
  https://aider.chat/docs/repomap.html
- **Repomix** (yamadashy, 23.7k stars): packs an entire repo into a
  single XML/MD/JSON file with tree-sitter compression. More of a
  one-shot dump; less suitable for incremental agent use.
  https://github.com/yamadashy/repomix
- **rendergit** (Karpathy, 2.2k stars): renders a repo into a single
  HTML page for LLM consumption. Simpler approach, no AST parsing.
  https://github.com/karpathy/rendergit

**Implementation for Norn:**
- tree-sitter parsing for Python and TypeScript (primary languages)
- Build map: `file_path → [(symbol_type, name, line_start, line_end, signature)]`
- Output format optimized for LLM consumption (ACI principle):
  ```
  src/norn/core/agent.py (23 symbols)
    class NornAgent(BaseAgent):              L12–L95
      async def run(self, user_input: str)   L60–L67
      async def _run_impl(self, ...)         L69–L95
    MAX_TOOL_ROUNDS = 10                     L10
  ```
- Incremental update: rebuild only files with mtime changes (same
  pattern as Phase 10 obsidian index)
- Cache map in `~/.norn/cache/repo_map.json` per project
- Inject condensed map into system prompt or as first context message
- On-demand expansion: when agent needs details, `file_read` with line
  range returns only the relevant function body

**Config:**
```yaml
context:
  repo_map: true               # enable/disable
  repo_map_languages:          # tree-sitter grammars to load
    - python
    - typescript
  repo_map_max_tokens: 2000    # cap map size
  repo_map_exclude:            # patterns to skip
    - "tests/**"
    - "docs/**"
    - ".venv/**"
```

**Files touched:** `src/norn/core/repo_map.py` (new),
`src/norn/core/agent.py`, `configs/default.yaml`.

**Dependencies:** `tree-sitter>=0.23`, `tree-sitter-python`,
`tree-sitter-typescript` as optional extras.

#### 4.8 OpenEvolve — Evolutionary Prompt & Config Optimization

**What:** Use evolutionary search (LLM-as-mutator) to automatically
optimize Norn's system prompts, tool schemas, and configuration
parameters. Instead of hand-tuning, let a population of prompt variants
compete on a fitness function (benchmark score, token efficiency, task
success rate).

**This is NOT a runtime optimization** — it's an offline meta-optimization
loop that produces better static artifacts (prompts, schemas, configs)
that Norn then uses at runtime.

**Savings:** Indirect but compounding. OpenEvolve has demonstrated +23%
accuracy on prompt optimization tasks (HotpotQA benchmark).

**Complexity:** Medium-High.

**Prior art:**
- **OpenEvolve** (Asankhaya Sharma, 6.1k stars): open-source
  implementation of Google DeepMind's AlphaEvolve. Uses MAP-Elites
  (quality-diversity grid) + island-based parallel populations + LLM
  ensemble for mutation. `pip install openevolve`.
  https://github.com/algorithmicsuperintelligence/openevolve
- **ADAS** (Hu, Lu, Clune, 2024, arXiv:2408.08435): Automated Design
  of Agentic Systems — meta-agent that programs new agent architectures.
- **Meta-Harness** (Stanford, March 2026): end-to-end optimization of
  model harnesses. Referenced in Norn architecture doc.

**What to evolve in Norn:**
1. **System prompt** — compress without quality loss, find optimal
   phrasing per model family
2. **Tool descriptions** — find minimal descriptions that preserve
   correct tool selection
3. **Config parameters** — `max_tool_rounds`, `temperature`,
   `max_tokens`, context window thresholds
4. **Tool selection heuristics** — keyword→tool mappings (see 4.5)

**Implementation:**
- Define a fitness function using Phase 11 benchmark:
  `score = task_success_rate * 100 - prompt_tokens * 0.001`
- Initial population: current prompts/configs + hand-written variants
- Mutation: OpenEvolve sends current prompt + fitness score to LLM,
  asks for improved variant
- Evaluation: run Phase 11 benchmark suite on each variant
- Selection: MAP-Elites keeps best per (model_family, task_category)
- Output: optimized prompt/config files committed to repo

**Integration pattern:**
```bash
# Offline optimization (not in agent runtime)
norn evolve --target system_prompt --generations 20 --population 10
norn evolve --target tool_schemas --generations 10
# Produces: configs/optimized/system_prompt_qwen3.txt, etc.
```

**Prerequisite:** Phase 11 benchmark (fitness function).

**Files touched:** `src/norn/cli/evolve.py` (new),
`src/norn/evolve/` (new module), `configs/optimized/` (output).

**Dependencies:** `openevolve>=0.2` as optional extra.

#### 4.9 LLMLingua-2 Compression

**What:** Microsoft's prompt compression library. Uses a small model
(XLM-RoBERTa) to identify and remove non-essential tokens from text
while preserving meaning.

**Savings:** 2–5x compression with minimal quality loss on natural language.

**Complexity:** Medium-High — requires `pip install llmlingua`, runs
inference through a small model.

**When to use:** Compress large tool outputs (file contents, long bash
results) before inserting into history. Do NOT compress tool schemas,
system prompts, or structured content.

**Implementation:**
```python
from llmlingua import PromptCompressor
compressor = PromptCompressor(model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank")
compressed = compressor.compress_prompt(text, rate=0.5)  # 50% compression
```

**Caveats:**
- Adds ~100ms latency per compression
- Not suitable for JSON/YAML/code (only natural language)
- Optional dependency like obsidian extras

**References:**
- Paper: arXiv:2403.12968 (LLMLingua-2, 2024)
- Library: `pip install llmlingua`

#### 4.10 OpenAI Compaction API

**What:** Server-side endpoint `/responses/compact` that shrinks
conversation history into an opaque encrypted compaction item. Carries
forward reasoning state in fewer tokens.

**Savings:** 50–80% on long contexts.

**Complexity:** Low implementation, but OpenAI-only (Responses API, not
Chat Completions).

**Status:** Monitor. Not actionable until Norn supports OpenAI Responses
API (currently uses Chat Completions via LiteLLM).

---

## 5. Implementation Roadmap

### Wave 1 — Quick Wins (1–2 sessions)

| Task | Technique | Estimated Savings |
|------|-----------|-------------------|
| W1.1 | Tool schema minification (4.1) | 30–60% tool tokens |
| W1.2 | Tool output truncation cap (4.2) | 30–60% tool results |
| W1.3 | System prompt audit & compression (4.3) | 20–50% system prompt |

**Validation:** Before/after token count on 5 representative prompts.
Use `litellm.token_counter()`.

### Wave 2 — Context Management (3–4 sessions)

| Task | Technique | Estimated Savings |
|------|-----------|-------------------|
| W2.1 | Context manager module (4.4) | Foundation |
| W2.2 | Sliding window + summarization (4.4) | 50–80% on long sessions |
| W2.3 | Dynamic tool selection — keyword heuristic (4.5) | 30–80% tool tokens |
| W2.4 | Intermediate tool call collapse (4.6) | 40–60% on agent loops |

**Validation:** Token usage tracking in observability (Phase 9 v2 already
logs `prompt_tokens`). Compare sessions before/after.

### Wave 3 — Advanced (3–5 sessions, after Phase 10/11)

| Task | Technique | Estimated Savings |
|------|-----------|-------------------|
| W3.1 | Repo map — tree-sitter AST context (4.7) | 70–90% file tokens |
| W3.2 | OpenEvolve — evolutionary prompt optimization (4.8) | +10–25% quality, indirect token savings |
| W3.3 | LLMLingua-2 optional compression (4.9) | 2–5x on text outputs |
| W3.4 | OpenAI Compaction API support (4.10) | 50–80% (OpenAI only) |
| W3.5 | Dynamic tool selection — embedding upgrade (4.5 opt 2) | Better relevance |

**Prerequisite:** Phase 10 (embedder), Phase 11 (benchmark for regression
testing and OpenEvolve fitness function).

---

## 6. Configuration Schema

```yaml
# configs/default.yaml — additions
context:
  # Sliding window
  max_history_tokens: 8000       # trigger summarization above this
  recent_turns_keep: 6           # always keep N most recent turns verbatim
  summary_max_tokens: 200        # max tokens for summary message

  # Tool output
  max_tool_result_tokens: 2000   # truncate tool results above this
  truncation_strategy: "bookend" # bookend | head | summarize

  # Dynamic tools
  dynamic_tools: false           # enable dynamic tool selection
  always_include_tools:          # core tools always sent
    - bash
    - file_read
    - file_edit
    - file_write
  max_tools_per_turn: 8

  # Schema minification
  minify_tool_schemas: true      # strip verbose descriptions
```

---

## 7. Metrics & Observability

All optimizations should be measurable via existing observability (Phase 9 v2):

| Metric | Source | Target |
|--------|--------|--------|
| `prompt_tokens` per turn | LiteLLM response | -40% vs baseline |
| `completion_tokens` per turn | LiteLLM response | No regression |
| `cache_hit_rate` | Phase 9 v2 G | Maintain or improve |
| `tools_sent` per turn | New: log in agent.py | Track tool selection |
| `context_summary_count` | New: log in context.py | Track summarizations |
| Task success rate | Phase 11 benchmark | No regression |

---

## 8. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Schema minification breaks tool use on some models | High | A/B test per model family; keep verbose mode as fallback |
| History summarization loses critical context | High | Pin mechanism for important messages; keep recent turns verbatim |
| Dynamic tool selection omits needed tool | Medium | `always_include_tools` core set; fallback to full set on tool-not-found error |
| LLMLingua corrupts code/JSON in tool output | Medium | Only compress natural language blocks; skip structured content |
| Compression latency overhead | Low | LLMLingua-2 is fast (~100ms); summarization async/cached |

---

## 9. Key References

### Papers
| Paper | Year | Key Contribution |
|-------|------|------------------|
| LLMLingua (arXiv:2310.05736) | 2023 | Perplexity-based token removal, up to 20x compression |
| LLMLingua-2 (arXiv:2403.12968) | 2024 | Data-distilled token classification, 3–6x faster than v1 |
| LongLLMLingua (arXiv:2310.06839) | 2023 | Optimized for long-context QA |
| Selective Context (arXiv:2310.06201) | 2023 | Self-information based sentence filtering |
| MInference (arXiv:2407.02490) | 2024 | KV-cache optimization for long context |
| SCBench (MS Research) | 2024 | Shared-context benchmark for evaluation |
| ADAS (arXiv:2408.08435) | 2024 | Meta-agent that programs new agent architectures |
| Meta-Harness (Stanford) | 2026 | End-to-end optimization of model harnesses |

### Provider Documentation
| Provider | Feature | URL |
|----------|---------|-----|
| OpenAI | Prompt Caching | https://platform.openai.com/docs/guides/prompt-caching |
| OpenAI | Compaction API | https://platform.openai.com/docs/guides/compaction |
| OpenAI | Tool Search | https://platform.openai.com/docs/guides/tools-tool-search |
| Anthropic | Prompt Caching | https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching |

### Libraries
| Library | Purpose |
|---------|---------|
| `llmlingua` (pip) | Prompt compression |
| `tiktoken` / `litellm.token_counter()` | Token counting |
| `tree-sitter` + `tree-sitter-python` + `tree-sitter-typescript` | AST parsing for repo maps |
| `openevolve` (pip) | Evolutionary prompt & config optimization |
| `repomix` (npm) | Full-repo packing for one-shot LLM context |

### Production Agent Patterns
| Agent | Technique Used |
|-------|----------------|
| Aider | Tree-sitter repo maps, chat summaries, diff-based context |
| Claude Code | Smart context eviction, ~100k window management |
| OpenCode | Skills system (conditional prompt injection) |
| Cursor | AST-based context selection |
| rendergit (Karpathy) | Full-repo HTML rendering for LLM consumption |

---

## 10. Estimated Effort

| Wave | Sessions | Dependencies |
|------|----------|-------------|
| Wave 1 (Quick Wins) | 1–2 | None |
| Wave 2 (Context Management) | 3–4 | None |
| Wave 3 (Advanced) | 3–5 | Phase 10 (embedder), Phase 11 (benchmark) |
| **Total** | **7–11 sessions** | |

---

## 11. Success Criteria

- [ ] Baseline token measurement established (current prompt_tokens per
      scenario type)
- [ ] Wave 1 achieves ≥30% prompt token reduction on one-shot tasks
- [ ] Wave 2 achieves ≥50% reduction on 10+ turn chat sessions
- [ ] Phase 11 benchmark shows no quality regression after each wave
- [ ] All optimizations are configurable and individually toggleable
