# Norn Phase 10 — Harness Engineering Design

**Date:** 2026-05-06
**Status:** Design complete, ready for implementation
**Prerequisite:** Phase 9 v2 (all workstreams complete) ✅
**Inspiration:** KIRA (TerminalBench harness), Meta-Harness paper (arXiv:2603.28052v1)

---

## 1. Problem Statement

Norn's current agent loop is functionally correct but **naive about context management**:

1. **No environment awareness** — the LLM wastes 2-4 turns on exploratory tool calls (`ls`, `git status`, `cat package.json`) at the start of every session. These are deterministic facts the harness could provide upfront.

2. **No per-turn output budget** — each tool call can inject up to 8000 chars. With 5 tool calls per turn, a single LLM round can accumulate ~40K chars of tool output. No global budget prevents context bloat within a turn.

3. **Full history every turn** — `agent.py:96-98` sends `[system_prompt, *history]` verbatim. On turn 20 with heavy tool use, this is 50-100K tokens. No sliding window, no summarization, no eviction.

4. **Shallow routing signals** — the router uses only 4 boolean signals (long prompt, keyword match, history length, tool count). No domain classification, no task-type awareness.

---

## 2. Goal

Implement 4 harness-level improvements that reduce wasted LLM calls, control context growth, and improve routing accuracy — all as **purely additive modifications** (lesson from Meta-Harness: 0 regressions from additive-only changes).

**Success criteria:**
- Env bootstrap eliminates ≥2 exploratory tool calls on typical first-turn
- Per-turn output budget caps at configurable threshold (default 30K chars)
- Sliding window keeps context under a configurable token budget
- Domain signals improve routing accuracy (measured via Phase 11 benchmark when available)

---

## 3. Feature 1: Environment Bootstrap

### Concept

Before the first LLM call in a session, scan the working directory and inject a structured `[Environment Snapshot]` block into the system prompt. This gives the model immediate awareness of:

- Git status (branch, clean/dirty, recent commits)
- Project type (languages detected, package managers, key config files)
- Directory structure (top-level layout)
- Runtime environment (Python version, venv, Node version if applicable)

### Design

**New module:** `src/norn/core/env_bootstrap.py`

```python
@dataclass
class EnvironmentSnapshot:
    """Structured project environment information."""
    cwd: str
    git_branch: str | None
    git_dirty: bool
    git_recent_commits: list[str]  # last 3 one-liners
    languages: list[str]           # detected: python, typescript, rust, etc.
    package_managers: list[str]    # uv, pip, npm, cargo, etc.
    key_files: list[str]           # pyproject.toml, package.json, Cargo.toml, etc.
    directory_tree: str            # top-2-level tree (abbreviated)
    python_version: str | None
    venv_active: bool

    def render(self) -> str:
        """Render as a compact text block for system prompt injection."""
        ...
```

**Detection logic** (all synchronous, fast filesystem checks):
- Git: `git rev-parse --abbrev-ref HEAD`, `git status --porcelain`, `git log --oneline -3`
- Languages: presence of `*.py`, `*.ts`, `*.rs`, `Cargo.toml`, `go.mod`, etc.
- Package managers: presence of `pyproject.toml` (uv/pip), `package.json` (npm/pnpm), `Cargo.toml` (cargo)
- Directory tree: `os.listdir()` top-level + 1 level deep for `src/`, `tests/`, `lib/`
- Python: `sys.version` if Python project
- Venv: check `VIRTUAL_ENV` env var or `.venv/` directory

**Injection point:** `agent.py:_build_system_prompt()` — append snapshot block after memory.

**Config:**
```yaml
agent:
  env_bootstrap: true  # enable/disable
```

**Constraints:**
- Must complete in <200ms (no network calls, no heavy computation)
- Output capped at 500 chars (compact rendering)
- Git commands use `subprocess.run(timeout=2)` — failure is silent (graceful degradation)

### Impact

- Eliminates 2-4 exploratory turns per session
- ~1 token saved per token that would have been in tool_call + tool_result for those turns
- Zero regression risk (purely additive information)

---

## 4. Feature 2: Per-Turn Output Budget

### Concept

Cap the total chars of tool output injected per LLM round. When multiple tool calls in a turn would exceed the budget, apply progressive truncation to keep total output within bounds.

### Design

**New module:** `src/norn/core/turn_budget.py`

```python
class TurnBudgetTracker:
    """Track and enforce per-turn output budget."""

    def __init__(self, max_chars_per_turn: int = 30_000):
        self.max_chars_per_turn = max_chars_per_turn
        self._chars_used = 0

    def reset(self) -> None:
        """Reset at the start of each LLM round."""
        self._chars_used = 0

    def allocate(self, raw_output: str, max_per_tool: int) -> str:
        """Allocate budget for a tool result, applying truncation if needed."""
        remaining = self.max_chars_per_turn - self._chars_used
        if remaining <= 0:
            return "[Output budget exhausted for this turn]"

        effective_max = min(max_per_tool, remaining)
        result = truncate_tool_output(raw_output, effective_max)
        self._chars_used += len(result)
        return result
```

**Integration in `agent.py`:**
- Create `TurnBudgetTracker` at start of `_run_impl()` / `run_stream()`
- Call `tracker.reset()` at the top of each `for _round` iteration
- Replace direct `truncate_tool_output()` calls with `tracker.allocate()`

**Config:**
```yaml
agent:
  max_turn_output_chars: 30000  # per-turn budget across all tool calls
  max_tool_result_chars: 8000   # per-tool cap (existing)
```

**Behavior:**
- First tool call gets full `min(8000, 30000)` budget
- Subsequent calls get progressively less if budget is depleting
- If budget is exhausted: tool result replaced with `[Output budget exhausted]`
- Budget resets each LLM round (not cumulative across rounds)

### Impact

- Prevents context explosion on multi-tool turns (5 calls × 8K = 40K → capped at 30K)
- Graceful degradation: most important tools are called first
- Simple, no LLM calls required

---

## 5. Feature 3: Sliding Window + Summarization

### Concept

Keep only the last N turns verbatim in context. When the total message history exceeds a token budget, older turns are evicted and replaced with a condensed summary. This is the single highest-impact optimization for long sessions.

### Design

**New module:** `src/norn/core/context.py`

```python
class ContextManager:
    """Manages conversation history with sliding window and summarization."""

    def __init__(
        self,
        max_history_tokens: int = 8000,
        recent_turns_keep: int = 6,
        summary_max_tokens: int = 300,
        summary_provider: LLMProvider | None = None,
    ):
        ...

    async def build_messages(
        self,
        system_prompt: str,
        history: list[Message],
    ) -> list[Message]:
        """Build the message list with window management.

        Returns: [system, summary_msg?, ...recent_turns]
        """
        ...

    def _count_tokens(self, messages: list[Message]) -> int:
        """Count tokens using litellm.token_counter or len-based estimate."""
        ...

    async def _summarize(self, old_messages: list[Message]) -> str:
        """Generate a condensed summary of evicted messages."""
        ...
```

**Architecture:**
```
[system_prompt]
[summary_of_evicted_turns]   ← injected when history exceeds budget
[recent_N_turns]             ← always kept verbatim
[tools]                      ← sent by LLM provider
```

**Token counting strategy:**
1. Primary: `litellm.token_counter(model, messages)` — accurate per-model
2. Fallback: `len(text) // 4` heuristic (if litellm unavailable)

**Summarization prompt:**
```
Summarize the conversation above in ≤{max_tokens} tokens.
Preserve: key decisions, file paths mentioned, task context, errors encountered.
Omit: verbose tool outputs, intermediate steps that led nowhere.
Format: dense bullet points.
```

**Trigger logic:**
- After each user message, before building `messages` list
- Check: `_count_tokens(history) > max_history_tokens`
- If over budget: summarize all but last `recent_turns_keep` turns
- Cache summary: only regenerate when new turns are evicted (not every turn)

**Pin mechanism (future):**
- Messages can be marked `pinned=True` (never evicted)
- Initial user message (task description) is auto-pinned
- Explicit pin via `/pin` slash command (deferred to later phase)

**Config:**
```yaml
context:
  sliding_window: true
  max_history_tokens: 8000
  recent_turns_keep: 6
  summary_max_tokens: 300
  summary_model: null  # null = use primary model
```

**Integration in `agent.py`:**
- Replace `messages = [Message(role=SYSTEM, ...), *self.history]` with:
  ```python
  messages = await self.context_manager.build_messages(system_prompt, self.history)
  ```
- `ContextManager` injected via `AgentLoop.__init__()` (optional, None = legacy behavior)

### Impact

- 50-80% token reduction on sessions >10 turns
- Enables much longer productive sessions before context degradation
- Summary quality depends on model capability (fast models = shorter summaries)

---

## 6. Feature 4: Domain-Aware Routing

### Concept

Enhance the router's complexity scoring with domain classification signals. The Meta-Harness paper showed that domain-aware strategies (routing to specialized BM25 retrieval per domain) significantly improve performance. For Norn, this means routing to POWERFUL tier for domains that benefit from stronger reasoning.

### Design

**Enhancement to:** `src/norn/core/router.py`

**New signals added to `_score_complexity()`:**

```python
# Domain patterns that benefit from POWERFUL tier
_DOMAIN_PATTERNS = {
    "architecture": re.compile(r"\b(?:system design|microservice|database schema|API design)\b", re.I),
    "security": re.compile(r"\b(?:vulnerabilit|injection|XSS|CSRF|auth bypass|CVE)\b", re.I),
    "ml_training": re.compile(r"\b(?:training loop|loss function|gradient|backprop|hyperparameter)\b", re.I),
    "debugging": re.compile(r"\b(?:stack trace|segfault|deadlock|race condition|memory leak)\b", re.I),
}

# Patterns that can safely use FAST tier
_SIMPLE_PATTERNS = {
    "simple_question": re.compile(r"^(?:what|how|where|when|why)\s.{5,80}\?$", re.I),
    "file_operation": re.compile(r"\b(?:read|write|create|delete|rename)\s+(?:file|dir)", re.I),
}
```

**Updated scoring:**
- Domain match (architecture, security, ml_training, debugging) → +2 to score
- Simple pattern match → -1 to score (floor at 0)
- Existing signals remain unchanged

**Observability:** log `domain_detected` field in routing decision event.

**Config:**
```yaml
router:
  domain_routing: true  # enable domain-aware signals
```

### Impact

- Better model allocation for complex tasks (security audits get POWERFUL)
- Simple questions route to FAST (cheaper, lower latency)
- Marginal improvement over current 4-signal approach
- Zero risk: only affects tier selection, not behavior

---

## 7. Implementation Sequencing

| # | Feature | Effort | Risk | Dependencies |
|---|---------|--------|------|--------------|
| 1 | Env Bootstrap | 1 session | None | None |
| 2 | Per-Turn Output Budget | 0.5 session | None | None |
| 3 | Sliding Window + Summarization | 2-3 sessions | Low | Token counting (litellm) |
| 4 | Domain-Aware Routing | 1 session | None | None |

**Total: 4.5-5.5 sessions**

Features 1, 2, and 4 are fully independent. Feature 3 depends on nothing but is larger and should be done last (or after 1+2 are stable).

---

## 8. Configuration Schema (Combined)

```yaml
# configs/default.yaml — additions for Phase 10
agent:
  env_bootstrap: true           # Feature 1
  max_turn_output_chars: 30000  # Feature 2
  max_tool_result_chars: 8000   # existing (per-tool cap)
  max_tool_rounds: 25           # existing

context:
  sliding_window: false         # Feature 3 (opt-in initially)
  max_history_tokens: 8000
  recent_turns_keep: 6
  summary_max_tokens: 300
  summary_model: null

router:
  domain_routing: true          # Feature 4
```

---

## 9. Files Touched (Summary)

| Feature | New Files | Modified Files |
|---------|-----------|----------------|
| Env Bootstrap | `src/norn/core/env_bootstrap.py` | `agent.py`, `config.py`, `configs/default.yaml` |
| Output Budget | `src/norn/core/turn_budget.py` | `agent.py`, `config.py`, `configs/default.yaml` |
| Sliding Window | `src/norn/core/context.py` | `agent.py`, `config.py`, `cli/main.py`, `configs/default.yaml` |
| Domain Routing | — | `router.py`, `config.py`, `configs/default.yaml` |

**Test files (all new):**
- `tests/test_core/test_env_bootstrap.py`
- `tests/test_core/test_turn_budget.py`
- `tests/test_core/test_context_manager.py`
- `tests/test_core/test_router_domain.py`

---

## 10. Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Env bootstrap subprocess hangs | Blocks agent start | 2s timeout on all subprocess calls; failure = skip gracefully |
| Output budget too aggressive | Model loses important info | Budget is per-turn not per-session; default 30K is generous |
| Summary loses critical context | Model forgets task details | Pin mechanism; always keep initial user message; recent_turns_keep=6 |
| Summarization LLM call adds latency | Slower first response after eviction | Only triggered when budget exceeded; cached until new eviction |
| Domain regex false positives | Wrong tier selected | Tier mismatch is self-correcting (fallback still works); log for tuning |

---

## 11. References

- KIRA (krafton-ai/KIRA): env bootstrap pattern, output cap (30KB default)
- Meta-Harness (arXiv:2603.28052v1): additive-only safety principle, domain routing
- Token Optimization Design (`docs/plans/2026-04-22-norn-token-optimization-design.md`): Wave 2 specs (§4.4-4.6) — this phase implements the core of Wave 2
- Aider: chat summaries pattern (inspiration for sliding window)
- Claude Code: smart context eviction (reference for pin mechanism)
