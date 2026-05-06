# Repo Map — AST-Based Context Design

**Date:** 2026-05-06
**Status:** Approved, ready for implementation
**Prerequisite:** Phase 10 Harness Engineering (env bootstrap wiring) ✅
**Reference:** Token Optimization Design §4.7

---

## 1. Problem Statement

When Norn starts working on a project, the LLM has no awareness of the codebase structure. It must issue multiple `file_read` and `grep` calls just to understand what exists. A single large file read injects 2000-5000 tokens of raw code when only the 200-token signature block was needed.

---

## 2. Goal

Generate a condensed map of the codebase — file paths with function/class signatures and line numbers — injected into the system prompt. The agent sees the "shape" of the code and requests specific function bodies on demand via targeted `file_read` with line ranges.

**Savings:** 70-90% vs sending full file contents. A 2000-token module becomes a ~200-token signature block.

---

## 3. Design

### Module: `src/norn/core/repo_map.py`

**Extraction strategy:**
- **Python:** `ast.parse()` (stdlib, 100% reliable, zero deps)
- **TypeScript/JavaScript:** regex-based extraction of `class`, `function`, `const`/`export` declarations
- **Other languages:** skip (extensible later)

### Output format

```
## Repo Map

src/norn/core/agent.py (5 symbols)
  class AgentLoop                           L39
    def __init__(self, llm, registry, ...)  L44
    async def run(self, user_input)         L83
    async def run_stream(self, user_input)  L157
  DEFAULT_MAX_TOOL_ROUNDS = 25              L42

src/norn/core/router.py (3 symbols)
  class RouterProvider                      L132
    async def complete(self, ...)           L161
  def _classify_complexity(msgs, tools)     L76
```

### Architecture

```
scan_repo(cwd, config) -> RepoMap
  ├─ discover_files(cwd, exclude_patterns, languages)
  ├─ for each file:
  │   ├─ check mtime vs cache → skip if unchanged
  │   ├─ parse_python(path) OR parse_typescript(path)
  │   └─ store FileSymbols(path, symbols, mtime)
  └─ render(file_symbols, max_chars) -> str
```

### Python extraction (`ast`)

Walk the AST, extract:
- `ClassDef` → name, line, bases
- `FunctionDef` / `AsyncFunctionDef` → name, line, args signature (truncated)
- Module-level `Assign` to UPPER_CASE names → constants
- Nested methods inside classes (1 level of nesting)

### TypeScript/JS extraction (regex)

Patterns:
```python
_TS_PATTERNS = [
    r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
    r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)",
    r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*(?::\s*\w+)?\s*=",
    r"^\s+(?:async\s+)?(\w+)\s*\([^)]*\)\s*[:{]",  # method inside class
]
```

### File discovery

- Walk `cwd` recursively
- Filter by language extensions: `.py`, `.ts`, `.tsx`, `.js`, `.jsx`
- Exclude patterns (glob): `.venv/**`, `node_modules/**`, `__pycache__/**`, `*.egg-info/**`, `dist/**`, `build/**`, `*_pb2.py`, `migrations/**`
- Skip files >1MB (truly enormous, likely generated)
- Skip hidden directories (`.git/`, `.tox/`, etc.)

### Rendering & Budget

- Global output cap: `max_chars` (default 2000)
- Priority: files closer to project root ranked higher
- Within a file: show all symbols (class > function > constant ordering)
- If over budget: truncate deepest files first, keep top-level structure
- Signature truncation: function args capped at 40 chars (`def foo(self, a, b, ...) -> R`)

### Caching

- In-memory dict: `{path: (mtime, list[Symbol])}`
- On scan: check `os.stat().st_mtime` — if unchanged, use cache
- Cache lives on the `RepoMap` instance (persists within a session)
- No disk persistence (rebuilt each session in <200ms for typical projects)

### Injection point

In `AgentLoop._build_system_prompt()`, after env snapshot:
```python
if self._repo_map:
    prompt += "\n\n" + self._repo_map
```

Generated once at AgentLoop construction (like env snapshot), not rebuilt per-turn.

---

## 4. Configuration

```yaml
agent:
  repo_map: true                 # enable/disable
  repo_map_max_chars: 2000       # output cap
  repo_map_languages:            # which languages to parse
    - python
    - typescript
  repo_map_exclude:              # glob patterns to skip
    - ".venv/**"
    - "node_modules/**"
    - "__pycache__/**"
    - "*.egg-info/**"
    - "dist/**"
    - "build/**"
    - "*_pb2.py"
    - "migrations/**"
```

---

## 5. Constraints

- Must complete in <500ms on a 500-file project
- Files >1MB skipped silently (likely generated)
- Syntax errors in individual files → skip that file, continue
- Zero external dependencies (ast is stdlib, regex is stdlib)
- Graceful degradation: if repo_map generation fails entirely, agent works normally without it

---

## 6. Data Model

```python
@dataclass
class Symbol:
    name: str
    kind: Literal["class", "function", "async_function", "constant", "method", "async_method"]
    line: int
    signature: str  # e.g. "def foo(self, x: int) -> str"
    children: list[Symbol]  # methods inside a class

@dataclass
class FileSymbols:
    path: str  # relative to cwd
    symbols: list[Symbol]
    mtime: float

class RepoMap:
    def __init__(self, config: RepoMapConfig): ...
    def scan(self, cwd: str) -> str:
        """Scan cwd and return rendered repo map string."""
    def _discover_files(self, cwd: str) -> list[Path]: ...
    def _parse_file(self, path: Path) -> FileSymbols | None: ...
    def _parse_python(self, path: Path, source: str) -> list[Symbol]: ...
    def _parse_typescript(self, path: Path, source: str) -> list[Symbol]: ...
    def _render(self, files: list[FileSymbols], max_chars: int) -> str: ...
```

---

## 7. Impact

- Eliminates 70-90% of token waste from full file reads
- Agent can make targeted `file_read` with line ranges instead of reading entire files
- Combined with env bootstrap: agent starts with both project structure AND code shape awareness
- Zero risk: purely additive, config-gated, no external deps
