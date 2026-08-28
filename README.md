# Norn

> The coding agent that weaves your destiny.

**Norn** is an open-source, model-agnostic coding agent with persistent memory,
multi-agent orchestration, and structured observability. Built in Python 3.11+
with a focus on MLOps-friendly development workflows.

**Status:** Active development (SOTA v2 — reliability, security, cost). Pre-1.0,
APIs and config schema may evolve.

**License:** MIT

---

## Why Norn?

Most coding agents are tied to a single vendor or hide their internals. Norn
is built on a different bet:

- **Model-agnostic** — any provider supported by [litellm][litellm] (OpenAI,
  Anthropic, OpenRouter, Ollama local, etc.) via a 3-tier router with
  automatic fallback.
- **Persistent memory** — sessions are logged, consolidated by a "dream"
  process, and reused as context across runs.
- **Multi-agent orchestration** — a Coordinator can decompose a task into
  parallel worker agents (research / build / verify phases).
- **Structured observability** — every tool call, permission decision, LLM
  request and cost is emitted as a JSONL event you can `tail`, query, or
  pipe into your own pipeline.
- **MLOps-aware tools** — built-in tools for inspecting datasets (Polars),
  models (PyTorch / scikit-learn), tensors, and writing model cards.
- **Permission system** — every destructive action is classified by risk and
  gated by a configurable mode (`interactive`, `auto`, `yolo`, `strict`).

[litellm]: https://github.com/BerriAI/litellm

---

## Quick start

### 1. Install

Norn uses [`uv`][uv] for package management.

```bash
git clone <your-fork-or-clone-url> norn
cd norn
uv sync --all-extras
```

[uv]: https://github.com/astral-sh/uv

### 2. Configure a provider

Create a `.env` at the project root with your API key (only needed for
hosted providers):

```bash
OPENROUTER_API_KEY=sk-or-...
# or
ANTHROPIC_API_KEY=sk-ant-...
# or
OPENAI_API_KEY=sk-...
```

The default config (`configs/default.yaml`) ships with a free OpenRouter
model (`stepfun/step-3.5-flash:free`). For a fully local setup, start
[Ollama][ollama] and switch the provider:

```yaml
# configs/default.yaml
llm:
  provider: "ollama"
  model: "qwen2.5-coder:14b"
```

[ollama]: https://ollama.ai

### 3. Run

```bash
# One-shot prompt
uv run norn run "list the python files in src/ and count the lines"

# Interactive chat
uv run norn chat

# Show available tools
uv run norn tools

# Show current config
uv run norn config

# Tail recent observability events (Phase 9 v1)
uv run norn logs tail
```

---

## Commands

| Command | Description |
|---|---|
| `norn chat` | Start an interactive chat session. |
| `norn run "<prompt>"` | Execute a one-shot prompt and exit. |
| `norn dream` | Manually trigger memory consolidation (requires `dream_system` flag). |
| `norn coordinate "<task>"` | Run a multi-agent orchestrated task (requires `coordinator` flag). |
| `norn tools` | List enabled tools with their risk level. |
| `norn config` | Show the current resolved configuration. |
| `norn logs tail` | Stream recent structured log events (filters: `--event`, `--session`, `--json`). |
| `norn version` | Print version. |

Most commands accept `--model {fast,standard,powerful}` (router mode) and
`--verbose` for DEBUG-level logging.

---

## Architecture

```
src/norn/
├── cli/                CLI entrypoints (Typer)
├── core/               Agent loop, LLM provider, router, config
├── tools/              Tool implementations
│   ├── bash, file_*, glob, grep
│   ├── ml/             dataset_inspector, model_*, tensor_inspector, model_card
│   └── web/            web_fetch, web_search
├── permissions/        Risk classifier + permission checker
├── memory/             Session logger + memory store
├── dream/              Memory consolidation engine
├── coordinator/        Multi-agent engine (research/build/verify phases)
├── observability/      Structured logging, processors, sinks
├── flags/              Feature flag registry
└── mcp/                MCP server adapters
```

### Key design choices

- **Tools return structured `ToolResult`** — never raw exceptions to the LLM.
  Failures carry a typed `error_type` (taxonomy in progress, Phase 9 v2).
- **Permission decisions are auditable** — every gate emits a log event with
  the classified risk, the mode, and the reason
  (`PermissionDecisionReason` StrEnum).
- **Observability is first-class** — `~/.norn/logs/*.jsonl` is the canonical
  audit trail; cost, tokens, latency are computed per turn and persisted.
- **Sessions are isolated** — each run gets a UUID; logs and memory are
  bound to it; `norn logs tail --session last` resolves the most recent.

---

## Configuration

Defaults live in `configs/default.yaml`. Environment overrides are applied
on load (see `NornConfig.apply_env_overrides`).

Highlights:

- **`permissions.mode`** — `interactive` (default, prompts user),
  `auto` (auto-approves low/medium risk), `yolo` (auto-approves everything),
  `strict` (denies destructive without explicit handler).
- **`flags.*`** — feature flags for `dream_system`, `coordinator`, `ml_tools`,
  `web_search`, `mcp`.
- **`router.enabled`** — when true, switches to 3-tier routing
  (`fast` / `standard` / `powerful`) with automatic technical-error fallback.
- **`logging.*`** — level, output (console / file / both), JSONL directory,
  redacted keys.

---

## Development

### Tests

```bash
uv run pytest                # full suite (~970 tests)
uv run pytest -k logger      # filter
uv run pytest --co           # collect only
```

### Lint

```bash
uv run ruff check .
uv run ruff check . --fix    # autofix safe issues
```

### Project conventions

- `from __future__ import annotations` everywhere
- Typer options use `Annotated[T, typer.Option(...)]`
- Type hints required on public APIs
- Conventional commits: `feat(scope):`, `fix(scope):`, `docs(scope):`,
  `chore:`, `test(scope):`, `refactor(scope):`, `perf(scope):`,
  `ci(scope):`
- Plans and design docs live in `docs/plans/YYYY-MM-DD-<topic>.md`

### Roadmap

Tracked in `docs/plans/`. Active plan: SOTA v2
(`docs/plans/2026-08-28-norn-sota-v2.md`) — wave 1 shipped: append-only
thread invariant + LLM record/replay tests, fail-closed bash sandbox
(seatbelt), byte-stable prompt + cache metrics, CodeRAG/judge/metrics wiring,
`norn bench guard` self-improvement safety net.

Backlog (see the SOTA v2 doc): AXI-style tool-output ergonomics, sliding
window / dynamic tools by default, Linux sandbox (bubblewrap/Landlock),
mypy in CI, Viktor code-writing paradigm (deferred).

---

## Contributing

This is a personal project, but issues and PRs are welcome. Before sending a
PR:

1. `uv run ruff check .` should not introduce new violations.
2. `uv run pytest` should be green.
3. Keep commits atomic and use the conventional format.
4. New features deserve a short design note in `docs/plans/`.

---

## License

MIT — see `LICENSE`.
