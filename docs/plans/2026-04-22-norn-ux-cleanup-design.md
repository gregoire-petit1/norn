# Norn — UX Cleanup: CLI Output & Logging

**Date:** 2026-04-22
**Status:** Approved, ready for implementation

---

## 1. Goal

Clean up terminal output so that normal usage shows only the response,
a dim metrics line, and dim tool-call progress — no structlog noise, no
LiteLLM INFO prints. Logs continue to `~/.norn/logs/` unchanged.

---

## 2. Target Rendering

### Normal mode (no flags)

```
> quelle est la capitale du brésil ?
⠦ Thinking...
La capitale du Brésil est Brasília.
  ⏱ 1.4s │ 1952→15 tokens │ qwen3-coder:480b-cloud
```

### With tool calls

```
> lis le fichier pyproject.toml et dis-moi la version
⠦ Thinking...
  [tool] file_read: pyproject.toml (0.1s)
⠦ Thinking...
La version actuelle est 0.1.0, définie dans pyproject.toml.
  ⏱ 2.8s │ 2340→42 tokens │ qwen3-coder:480b-cloud
```

### With --verbose

Unchanged: all structlog console logs visible (DEBUG level).

---

## 3. Changes

### 3.1 Suppress console logs in normal mode

**File:** `src/norn/observability/logger.py`

In normal mode (`output: "both"` or `output: "console"`), set the console
StreamHandler level to WARNING instead of following the global config level.
File sink stays at INFO. `--verbose` overrides console to DEBUG as today.

### 3.2 Suppress LiteLLM INFO prints

**File:** `src/norn/core/llm.py`

Add in `LiteLLMProvider.__init__()`:
```python
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
```

The existing `litellm.suppress_debug_info = True` stays.

### 3.3 Surface metrics in LLMResponse

**File:** `src/norn/core/agent.py` (and/or `src/norn/core/llm.py`)

Ensure the response object returned to the CLI carries:
- `latency_ms: int`
- `prompt_tokens: int`
- `completion_tokens: int`
- `model: str`

These are already logged; they just need to be accessible from the CLI layer.

### 3.4 Print dim metrics line after response

**File:** `src/norn/cli/main.py`

After `console.print(Markdown(response.content))`, print:
```python
console.print(
    f"  ⏱ {latency_s:.1f}s │ {prompt_tokens}→{completion_tokens} tokens │ {model}",
    style="dim",
)
```

Apply to both `norn run` (one-shot) and `norn chat` (REPL).

### 3.5 Print dim tool-call lines

**File:** `src/norn/cli/main.py` (or `src/norn/core/agent.py` via callback)

During the agent tool loop, emit a dim line per tool call:
```
  [tool] bash: ls -la (0.3s)
  [tool] file_read: pyproject.toml (0.1s)
```

This requires either:
- A callback/hook from the agent loop to the CLI, or
- The agent loop prints directly (simpler but couples CLI to agent)

Recommended: simple callback pattern — the CLI passes a callable to
`agent.run()` that gets called per tool execution.

---

## 4. Not Changed

- Rich spinner ("Thinking...") — already good
- Rich Markdown response rendering — already good
- File logging (JSONL in ~/.norn/logs/) — continues capturing everything
- `--verbose` flag — preserves current full-log behavior
- Chat mode startup banner — unchanged
- Config schema — no new fields needed

---

## 5. Estimated Effort

1 session. ~6 files touched, no new dependencies.
