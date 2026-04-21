# Norn — Phase 10 Design : `obsidian_rag` tool

**Date:** 2026-04-22
**Status:** Approved, ready for implementation planning
**Branch:** `chore/planning-future-phases`
**Related skills (existing):** `obsidian-cli`, `obsidian-markdown`, `obsidian-bases`,
`json-canvas` — to be cross-referenced during implementation.

---

## 1. Goal

Give Norn semantic access to the user's Obsidian vault for two complementary
use cases:

1. **Knowledge retrieval (read)** — the agent queries the vault for
   relevant context when answering dev questions ("what did I decide for
   the engie-scada MLOps stack?", "what patterns do I have for skill
   orchestration?").
2. **Session capture (write)** — the agent persists session notes,
   learnings, and decisions back into the vault, complementary to Norn's
   existing Dream memory system.

The tool is **opt-in** behind a feature flag and ships its own optional
dependency extras to keep Norn's default install lightweight.

---

## 2. Out of scope

YAGNI cuts:

- Interactive "ask my vault" chat command (Norn's agent loop already covers
  this through tool calls).
- Bidirectional Obsidian sync.
- Live filesystem watching (FSEvents/watchdog daemon) — see Phase 12 candidate.
- Cross-encoder re-ranking (YAGNI for ~80 docs).
- Hybrid BM25 + vector retrieval (YAGNI for MVP).
- Multilingual embedding model upgrade (MiniLM is acceptable for MVP).
- Wikilink resolution (following `[[other-note]]` to inject linked
  content) — strip-only for MVP.

---

## 3. Vault profile (target user: project owner)

Empirical context from inspection of `~/obsidian/myvault`:

- ~83 `.md` files, 1.3 MB total, ~13 500 lines
- Two main areas: `OpenCode/` (meta-project: patterns, decisions, memory,
  projects/engie-scada, stack, snippets) and `IADATA708-Pokec-Fairness/`
  (academic ML project)
- ~76% of files carry YAML frontmatter
- Mixed FR/EN content (FR dominant in personal notes, EN in technical
  patterns)
- Living vault, ~17 files modified in the last ~2 months — moderate
  activity, not high-frequency

Implication: a small RAG (~80–150 documents, 200–400 chunks) is
sufficient. Avoid over-engineering.

---

## 4. Architecture

```
┌─────────────────────────────────────────────────────────┐
│  LLM-facing tools (registered in ToolRegistry)            │
│  ┌──────────────────────┐   ┌──────────────────────────┐ │
│  │ obsidian_search        │   │ obsidian_write             │ │
│  │ (query, top_k=5)       │   │ (path, content, mode)      │ │
│  │ → list[Hit]            │   │ → ToolResult                │ │
│  └──────────┬─────────────┘   └────────────┬─────────────┘ │
└─────────────┼─────────────────────────────────┼─────────────┘
              │                                 │
┌─────────────┼─────────────────────────────────┼─────────────┐
│  Service layer                                                │
│  ┌──────────▼─────────────────────────────────▼─────────┐    │
│  │ ObsidianVault                                          │    │
│  │ - search(query, top_k) → list[Hit]                     │    │
│  │ - write(path, content, mode) → WriteResult             │    │
│  │ - reindex_incremental() (called once at norn boot)     │    │
│  │ - reindex_full() (called via `norn obsidian index`)    │    │
│  └─────────────┬───────────────────────────────────────────┘    │
└────────────────┼────────────────────────────────────────────────┘
                 │
┌────────────────┼────────────────────────────────────────────────┐
│  Indexing pipeline                                                │
│  ┌─────────────▼────┐  ┌──────────────────┐  ┌───────────────┐  │
│  │ MarkdownChunker   │→ │ Embedder          │→ │ VectorIndex    │  │
│  │ - parse YAML       │  │ - MiniLM-L6-v2    │  │ - sqlite +      │  │
│  │ - split H1/H2/H3   │  │ - CPU, 384-dim    │  │   sqlite-vec    │  │
│  │ - strip wikilinks  │  │ - cached locally  │  │ - mtime tracking│  │
│  │ - emit Chunk       │  │                   │  │ - knn (cosine)  │  │
│  └────────────────────┘  └──────────────────┘  └─────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. LLM-facing API

### 5.1 `obsidian_search`

```python
class ObsidianSearchInput(BaseModel):
    query: str = Field(..., description="Natural-language search query.")
    top_k: int = Field(5, ge=1, le=20, description="Number of hits to return.")


class Hit(BaseModel):
    path: str             # vault-relative path, e.g. "OpenCode/patterns/skills-guide.md"
    header_path: str      # H1 > H2 > H3 trail of the chunk
    snippet: str          # first ~300 chars of the chunk content
    score: float          # cosine similarity, [0, 1]
    tags: list[str]       # frontmatter tags
```

**Risk level:** `low` (read-only).

**Output formatting (markdown):**

```
Found 3 results for "MLOps decisions on engie-scada":

1. **OpenCode/projects/engie-scada/adrs/0003-orchestrator.md** (score: 0.84)
   `# ADR 0003 > ## Decision`
   We chose Prefect over Airflow because the team already runs Prefect
   in production for the upstream pipeline...

2. **OpenCode/projects/engie-scada/adrs/0001-stack.md** (score: 0.71)
   `# ADR 0001 > ## Stack choice`
   ...

3. **OpenCode/decisions/2026-03-15-mlflow.md** (score: 0.65)
   ...
```

### 5.2 `obsidian_write`

```python
class ObsidianWriteMode(StrEnum):
    CREATE = "create"   # fail if file exists
    APPEND = "append"   # append to existing file (or create if missing)
    OVERWRITE = "overwrite"  # explicit, requires permission re-confirm


class ObsidianWriteInput(BaseModel):
    path: str = Field(..., description="Vault-relative path, e.g. 'sessions/2026-04-22.md'.")
    content: str = Field(..., description="Markdown content to write.")
    mode: ObsidianWriteMode = ObsidianWriteMode.CREATE
```

**Risk level:** `medium` (file write under a controlled but user-owned path).

**Behaviour:**
- Resolve absolute path; reject if the resolved path is not under
  `vault_path` (path traversal protection).
- Atomic write (`tmp + rename`).
- Trigger background incremental reindex of the touched file (non-blocking).
- Returns: `ToolResult(output=f"wrote {len(content)} bytes to {rel_path}")`.

---

## 6. Indexing pipeline

### 6.1 MarkdownChunker

- Parse YAML frontmatter (`python-frontmatter`); store as `metadata: dict`.
- Strip the frontmatter from body before chunking.
- Split the body on H1/H2/H3 headers, preserving header trail
  (e.g. `# Patterns > ## Skills > ### Orchestration`).
- For each section: strip wikilinks `[[note]]` → `note` and
  `[[note|alias]]` → `alias`. Embeds `![[file]]` → empty.
- If a section exceeds ~800 tokens, split with a 50-token overlap
  (paragraph boundaries preferred).
- Frontmatter (tags + key fields) is **prepended** to each chunk's
  embedding text as context (e.g. `[tags: mlops, decision]\n\n<chunk>`).

Output: `Chunk(id, path, header_path, content, mtime, tags, metadata)`.

### 6.2 Embedder

- Wrapper around `sentence-transformers` with model
  `sentence-transformers/all-MiniLM-L6-v2` (90 MB, 384 dim).
- Lazy load on first use; cache to `~/.norn/obsidian/models`.
- CPU inference; batch encoding for index, single-shot for queries.
- Pure dependency injection (interface `Embedder.encode(texts) -> ndarray`)
  so a future `OpenAIEmbedder` or `VoyageEmbedder` can be plugged.

### 6.3 VectorIndex

- SQLite database at `~/.norn/obsidian/index.sqlite`.
- Loaded `sqlite-vec` extension at connection time.
- Schema:
  - `chunks(id INTEGER PK, path TEXT, header_path TEXT, content TEXT, mtime REAL, tags_json TEXT, metadata_json TEXT)`
  - `chunk_vec` virtual table from `sqlite-vec`: `(chunk_id INTEGER PK, embedding FLOAT[384])`
  - `files(path TEXT PK, mtime REAL, indexed_at REAL, chunk_count INTEGER)`
- API:
  - `upsert_chunks(file_path, mtime, chunks)` — atomic per-file replace.
  - `delete_file(path)` — for removed files.
  - `knn(query_vec, top_k)` → `list[(chunk_id, score)]`.
  - `stale_files(vault_paths_with_mtimes)` → list of paths to (re)index.

### 6.4 Incremental reindex

Algorithm (pseudo-code):

```python
def reindex_incremental():
    fs_files = {p: stat(p).st_mtime for p in walk(vault_path) if p.endswith(".md")}
    indexed = vector_index.list_files()  # dict[path, indexed_mtime]

    to_reindex = [p for p, m in fs_files.items() if m > indexed.get(p, 0)]
    to_delete = [p for p in indexed if p not in fs_files]

    for path in to_reindex:
        chunks = chunker.chunk_file(path)
        embeddings = embedder.encode([c.content for c in chunks])
        vector_index.upsert_chunks(path, fs_files[path], chunks, embeddings)

    for path in to_delete:
        vector_index.delete_file(path)

    log_event("obsidian.reindex", reindexed=len(to_reindex), deleted=len(to_delete))
```

Triggered:
- Once at `norn` CLI startup (when `flags.obsidian_rag` is true).
- Manually via `norn obsidian index` (full force-reindex via
  `--full` flag).

---

## 7. CLI commands

New `obsidian` Typer sub-app mounted under `norn`:

| Command | Behaviour |
|---|---|
| `norn obsidian index` | Run incremental reindex with progress bar. `--full` forces full reindex (drops index, rebuilds). |
| `norn obsidian status` | Print: vault path, total files, indexed files, total chunks, embedder model, index size on disk, last reindex timestamp. |
| `norn obsidian search "<query>"` | One-shot CLI search (no LLM); useful for debugging the index. Returns top-5 hits. |

---

## 8. Configuration

Additions to `configs/default.yaml`:

```yaml
flags:
  obsidian_rag: false   # Phase 10 — opt-in, requires `uv sync --extra obsidian`

obsidian:
  vault_path: "~/obsidian/myvault"
  index_path: "~/.norn/obsidian/index.sqlite"
  embedder:
    model: "sentence-transformers/all-MiniLM-L6-v2"
    cache_dir: "~/.norn/obsidian/models"
  chunker:
    max_tokens_per_chunk: 800
    overlap_tokens: 50
  reindex_on_boot: true
  search:
    top_k_default: 5
    top_k_max: 20
  write:
    enabled: true   # if false, only obsidian_search is registered
```

Pydantic config model gets a new `ObsidianConfig` block with the same
shape, plumbed through `NornConfig`.

---

## 9. Permissions

- `obsidian_search` → `RiskLevel.LOW` (read-only, no side effects).
- `obsidian_write` → `RiskLevel.MEDIUM` (file write, user-owned path, but
  potentially destructive in `overwrite` mode).
- Path-traversal defense: every write resolves the target via
  `Path(vault_path).resolve() / Path(rel_path)`; the resolved absolute
  path must satisfy `is_relative_to(vault_path.resolve())`. Otherwise
  → `ToolResult(error="path escapes vault", error_type=ToolErrorType.PERMISSION_DENIED.value)`.
- `overwrite` mode: classified as `RiskLevel.HIGH` to force prompt even
  in `auto` mode (only `yolo` skips).

---

## 10. Dependencies

`pyproject.toml` gains an optional extras block:

```toml
[project.optional-dependencies]
obsidian = [
    "sentence-transformers>=3.0",
    "sqlite-vec>=0.1.0",
    "python-frontmatter>=1.1",
    "numpy>=1.26",        # already pulled by torch in `ml` extra; explicit here for clarity
]
```

Install: `uv sync --extra obsidian`. The `obsidian_rag` flag refuses to
activate (with a clear error message pointing to `--extra obsidian`) if
the imports fail.

---

## 11. Observability

New event names (in `EventName`):

- `obsidian.reindex` — fields: `reindexed_count`, `deleted_count`,
  `total_chunks`, `latency_ms`.
- `obsidian.search` — fields: `query`, `top_k`, `hit_count`, `top_score`,
  `latency_ms` (already covered by tool.call but useful for direct
  service-level introspection if the tool layer is bypassed).
- `obsidian.write` — fields: `path`, `mode`, `bytes_written`,
  `triggered_reindex`.

The existing `tool.call` instrumentation already wraps the tool layer; the
events above are emitted at the service layer (one level deeper) for
finer attribution.

---

## 12. Tests

| Layer | Tests | Approach |
|---|---|---|
| Chunker | ~8 | YAML extract, H1/H2/H3 split, wikilink strip, embed strip, oversized section split, no-headers fallback, empty file, non-md skipped |
| Embedder | ~3 | Mock model returning deterministic vectors, batch vs single, lazy load |
| VectorIndex | ~6 | upsert + knn, replace on re-upsert, delete_file, stale_files detection, schema migration safety, empty index knn |
| Vault service | ~5 | search returns hits sorted by score, reindex_incremental skips unchanged, reindex_incremental detects deletions, write under vault OK, write outside vault rejected |
| Tools | ~6 | search returns formatted markdown, write create/append/overwrite, write rejects path traversal, search respects top_k bounds, error_type taxonomy applied |
| CLI | ~4 | `obsidian index`, `obsidian status`, `obsidian search` smoke tests + error when flag disabled |
| Integration / E2E | ~3 | Fixture vault (~5 .md files), full pipeline: reindex → search → write → reindex picks up new file |

Total estimate: **+30 to +35 tests**.

Test fixtures: `tests/fixtures/obsidian_vault/` with 5–7 small `.md`
files covering the chunker edge cases.

---

## 13. Risks & mitigations

| Risk | Mitigation |
|---|---|
| `sentence-transformers` install pulls torch (~500 MB) | Optional `obsidian` extra; clear error message with install hint |
| First boot slow (model download + initial index) | Progress feedback, persistent cache dir, async background reindex when possible |
| Vault grows past ~1 000 files | Linear-in-mtime incremental reindex stays fine; optional batch flag for large vaults |
| Path-traversal exploit (`../../etc/passwd`) | `resolve()` + `is_relative_to(vault_path.resolve())` checks, unit-tested |
| `sqlite-vec` API drift between minor versions | Pin `>=0.1.0,<0.2`; document upgrade path |
| Agent writes hostile / nonsense content into vault | Permission `medium` → user confirms in `interactive` mode; never auto in `auto` |
| Frontmatter tags collide with internal metadata keys | Namespace stored fields (`metadata_obsidian`) to avoid collision |
| French content quality with English-only MiniLM | Acceptable for MVP; document `multilingual-e5-base` as an upgrade path |
| Concurrent `norn` processes corrupt SQLite | SQLite WAL mode + `BEGIN IMMEDIATE` on writes; rare in single-user workflow but cheap insurance |
| Embedding model drift (model file changes) | Store `model_id` in `files` table metadata; force full reindex if mismatch |

---

## 14. Effort estimate

5 sessions of focused work:

1. **Session 1** — Chunker + Embedder + VectorIndex (TDD, pure logic, no IO heavy work)
2. **Session 2** — ObsidianVault service + reindex_incremental + SQLite plumbing
3. **Session 3** — `ObsidianSearchTool` + `ObsidianWriteTool` + permissions + ToolErrorType integration
4. **Session 4** — CLI sub-app (`obsidian index`, `status`, `search`) + observability events + config
5. **Session 5** — E2E tests with fixture vault + docs + polish

Each session ends with `pytest` + `ruff check` green and a clean commit
sequence. Conventional commits as in Phase 9 (`feat(obsidian):`,
`test(obsidian):`, etc.).

---

## 15. Success criteria

- `uv sync --extra obsidian` installs cleanly on a fresh checkout
- Enabling `flags.obsidian_rag: true` and running `norn obsidian index`
  produces a populated `~/.norn/obsidian/index.sqlite` in < 30 s
- `norn obsidian search "MLOps"` returns at least one relevant hit from
  the user's vault
- `norn run "search my obsidian vault for the engie-scada orchestrator decision"`
  triggers the `obsidian_search` tool and produces a coherent answer
  citing the actual ADR file
- `norn run "create a session note at sessions/test.md with body 'hello'"`
  creates the file under the vault, refuses if path escapes the vault
- All new tests pass (~30+); ruff baseline does not regress

---

## 16. Open questions (to revisit during implementation)

1. **Reindex on boot vs lazy** — boot reindex is cheap on 80 files but
   adds ~500 ms startup cost when the index is fresh and nothing
   changed. Consider a quick "any-mtime-newer-than-index?" probe before
   walking the full tree.
2. **Wikilink resolution** — explicitly out of scope for MVP, but
   surface in roadmap if user feedback shows the agent missing context.
3. **Backup before overwrite** — when `mode=overwrite`, write the
   previous content to `~/.norn/obsidian/backups/<path>.<ts>.md` first.
   Cheap insurance; trivial to add. Decide during implementation.
4. **`obsidian-cli` skill cross-reference** — the user already has an
   `obsidian-cli` skill loaded for vault interaction. Verify there is no
   functional overlap or that the new tool subsumes it cleanly.
5. **Search response formatting** — markdown table vs bullet list vs
   collapsible details. Default to bullet list; revisit after dogfooding.
6. **`reindex_on_boot` cost when disabled offline** — if the user runs
   `norn` without network and `sentence-transformers` model is not yet
   cached, boot fails. Detect and emit a clear "run `norn obsidian
   index` once with network" message.

---

## 17. Approval

Design approved by user (FR session, 2026-04-22). Branch:
`chore/planning-future-phases` (sibling worktree at
`~/norn-planning`, base `5a4da3b`).

Next step: write the detailed implementation plan in
`docs/plans/2026-04-22-norn-phase10-obsidian-rag-implementation.md` (in
a later iteration; for now this design is one of four planning docs being
written in parallel to the active Phase 9 v2 implementation).
