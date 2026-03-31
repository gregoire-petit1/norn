# Norn - Architecture Design

**Date**: 2026-03-31
**Status**: Approved
**Author**: gregoirepetit

---

## Vision

Open-source, model-agnostic, self-hosted coding agent inspired by Claude Code's architectural patterns (leaked 2026-03-31) and Hermes Agent's open philosophy. ML/MLOps-specialized, with persistent memory, multi-agent orchestration, and sophisticated permission system.

**Name**: Norn (Norse mythology -- the three weavers of destiny: Urd=past/memory, Verdandi=present/agent, Skuld=future/dream)

## Goals

1. **Full architectural control** -- own every layer, no compromises
2. **ML/MLOps focus** -- native PyTorch/sklearn/Polars integration
3. **Self-hosting sovereignty** -- local models, private data, no SaaS dependency
4. **Learning/research** -- understand agent construction from A to Z

## Stack

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Language | Python 3.11+ | MLOps ecosystem, fast POC, Hermes precedent |
| Package manager | uv | Fast, modern, mandatory per project standards |
| LLM inference | Hybrid local + API | Ollama/vLLM local default, OpenRouter/Anthropic/OpenAI fallback |
| CLI framework | click/typer + rich | Streaming markdown, no heavy TUI for POC |
| Async | asyncio | I/O-bound workload (LLM latency is the bottleneck) |
| Schemas | Pydantic v2 | Tool input validation, config models |
| Config | YAML | User-facing configuration |
| License | MIT | Open-source, no restrictions |

## Architecture Overview

```
+-----------------------------------------------------------+
|                        CLI (TUI)                          |
|                  Rich / Typer / Streaming                 |
+-----------------------------------------------------------+
|                     Agent Core Loop                       |
|         user input -> LLM -> tool calls -> repeat         |
+----------+----------+-----------+-------------------------+
| Tool     | Permission| Feature   | LLM Provider           |
| Registry | System    | Flags     | Abstraction            |
+----------+----------+-----------+-------------------------+
|                    Memory Layer                           |
|         memdir/ + MEMORY.md + session store               |
+-----------------------------------------------------------+
|                  Dream Engine                             |
|       3-gate trigger + 4-phase consolidation              |
+-----------------------------------------------------------+
|                  Coordinator                              |
|    Worker pool + scratchpad + R->S->I->V pipeline         |
+-----------------------------------------------------------+
```

## Project Structure

```
norn/
├── src/norn/
│   ├── core/           # agent.py, llm.py, config.py, models.py
│   ├── tools/          # base.py, registry.py, bash.py, file_*.py, glob.py, grep.py
│   ├── permissions/    # classifier.py, modes.py, protected.py
│   ├── memory/         # memdir.py, session.py, index.py
│   ├── dream/          # engine.py, trigger.py, prompts.py
│   ├── coordinator/    # coordinator.py, worker.py, scratchpad.py
│   ├── cli/            # main.py, commands.py
│   └── flags/          # registry.py
├── configs/            # default.yaml
├── tests/
│   ├── test_core/
│   ├── test_tools/
│   ├── test_dream/
│   └── test_coordinator/
├── docs/
│   └── plans/
├── pyproject.toml
└── README.md
```

---

## Phase 1: Core Agent Loop + Tools (Weeks 1-2)

### LLM Provider Abstraction

```python
class LLMProvider(Protocol):
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
    ) -> AsyncIterator[StreamChunk]: ...
```

**Implementations**:
- `OllamaProvider` -- local inference via HTTP API
- `LiteLLMProvider` -- universal fallback (OpenRouter, OpenAI, Anthropic, vLLM)
- `RouterProvider` -- smart routing based on task complexity (simple -> local small, complex -> API large)

### Tool System

```python
class RiskLevel(Enum):
    LOW = "low"       # read-only ops (grep, glob, read)
    MEDIUM = "medium"  # writes with undo (file edit)
    HIGH = "high"      # destructive (rm, git push force, bash)

class Tool(Protocol):
    name: str
    description: str
    risk_level: RiskLevel
    input_model: type[BaseModel]

    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult: ...
```

**Phase 1 tools (minimum viable)**:

| Tool | Risk | Description |
|------|------|-------------|
| BashTool | HIGH | Shell execution |
| FileReadTool | LOW | Read file/directory |
| FileWriteTool | MEDIUM | Create/overwrite file |
| FileEditTool | MEDIUM | Targeted replacement in file |
| GlobTool | LOW | File pattern matching |
| GrepTool | LOW | Content search (ripgrep) |

**Tool Registry** with cached JSON schemas for LLM prompt efficiency.

### Agent Core Loop

```python
class AgentLoop:
    async def run(self, user_input: str) -> None:
        messages = self.history + [user_message(user_input)]
        while True:
            response = await self.llm.complete(messages=messages, tools=self.registry.get_schemas())
            if not response.tool_calls:
                self.display(response.content)
                break
            for call in response.tool_calls:
                tool = self.registry.get(call.name)
                permission = await self.permissions.check(tool, call.input)
                if permission.denied:
                    result = ToolResult(error=permission.reason)
                else:
                    result = await tool.execute(call.input, self.ctx)
                messages.append(tool_result_message(call.id, result))
            messages.append(assistant_message(response))
```

### CLI

```
$ norn chat              # interactive mode
$ norn run "fix the bug" # one-shot
$ norn tools             # list available tools
$ norn config            # configure LLM provider
```

---

## Phase 2: Permission System + Feature Flags (Week 3)

### Permission Modes

| Mode | Behavior | Usage |
|------|----------|-------|
| interactive | Prompt user for MEDIUM/HIGH | Default |
| auto | Auto-approve LOW+MEDIUM, prompt HIGH | Fast workflow |
| yolo | Approve all without prompt | Dev/test only |
| strict | Prompt for ALL, even LOW | Sensitive env |

### Permission Classifier

Contextual risk escalation:
- Destructive bash patterns (`rm -rf`, `DROP TABLE`, `git push --force`) -> HIGH
- Protected paths (`.gitconfig`, `.bashrc`, `.env`, `.ssh/`) -> HIGH
- Path traversal prevention (URL-encoded, Unicode normalization, backslash injection)

### Feature Flags

Runtime flags with priority: env var > config file > default.

```python
KNOWN_FLAGS = {
    "dream_system": FeatureFlag("dream_system", True, "Memory consolidation"),
    "coordinator": FeatureFlag("coordinator", False, "Multi-agent mode"),
    "ml_tools": FeatureFlag("ml_tools", True, "MLOps-specific tools"),
    "voice_input": FeatureFlag("voice_input", False, "Voice transcription"),
}
```

Conditional imports: modules only loaded when their flag is enabled.

---

## Phase 3: Dream System (Week 4)

### Memory Directory

```
~/.norn/
├── memory/
│   ├── MEMORY.md              # Main index (< 200 lines, < 25KB)
│   ├── topics/
│   │   ├── project-foo.md     # Per-project memory
│   │   ├── user-prefs.md      # User preferences
│   │   └── ml-experiments.md  # Ongoing ML experiments
│   └── daily/
│       ├── 2026-03-31.md      # Daily log (append-only)
│       └── 2026-03-30.md
├── sessions/
│   ├── session-abc123.jsonl   # Session history
│   └── session-def456.jsonl
└── config.yaml
```

### Three-Gate Trigger

All three must pass before a dream runs:

1. **Time gate**: 24 hours since last dream
2. **Session gate**: >= 5 sessions since last dream
3. **Lock gate**: file lock acquired (prevents concurrent dreams)

### Four-Phase Consolidation

1. **Orient**: Read memory directory, MEMORY.md, existing topic files
2. **Gather Recent Signal**: Find new information worth persisting (daily logs -> session transcripts -> drifted memories)
3. **Consolidate**: Write/update memory files. Convert relative dates to absolute. Delete contradicted facts.
4. **Prune and Index**: Keep MEMORY.md under 200 lines / ~25KB. Remove stale pointers. Resolve contradictions.

The dream subagent gets **read-only** filesystem access. It can only write to `~/.norn/memory/`.

**Model choice**: Configurable. A small local model (Mistral 7B, Qwen 7B) suffices for consolidation -- no need for a large model.

### Session Integration

MEMORY.md is injected into the system prompt at session start. Dream is triggered as a background task (fire-and-forget).

---

## Phase 4: Coordinator Multi-Agent (Weeks 5-6)

### Pipeline

```
Phase 1: RESEARCH       (parallel workers investigate)
Phase 2: SYNTHESIS       (coordinator reads findings, crafts specs)
Phase 3: IMPLEMENTATION  (parallel workers implement per spec)
Phase 4: VERIFICATION    (parallel workers test + validate)
```

### Workers

Each worker is an isolated `AgentLoop` with:
- Its own LLM context
- A scoped subset of tools
- Write access only to its scratchpad section

### Scratchpad

```
/tmp/norn-coord-{id}/
├── research/
│   ├── worker-a-findings.md
│   └── worker-b-findings.md
├── specs/
│   ├── spec-1-refactor-auth.md
│   └── spec-2-add-tests.md
└── verification/
    └── worker-e-report.md
```

Coordinator reads all. Workers write only to their section. No direct inter-worker communication (can be added later via task-notification messages).

### Activation Heuristic

Coordinator only activates when justified (adds latency):
- Task text > 500 chars
- Keywords: "refactor", "across", "multiple files"
- Files in scope > 10
- Threshold: >= 2 indicators

### Coordinator Prompt Rules

- "Parallelism is your superpower"
- "Do NOT say 'based on your findings' -- read the actual findings"
- Coordinator synthesizes, workers execute

---

## Key Design Decisions

1. **Python over Rust for POC**: LLM latency is the bottleneck, not runtime. Rust can be introduced for hot paths later.
2. **Protocol over inheritance**: Tools, LLM providers use Python Protocols, not ABC. Composition over inheritance.
3. **asyncio everywhere**: All I/O is async. The agent loop, tools, dream, coordinator all use async/await.
4. **Memory as files, not DB**: Plain markdown files for memory (like Claude Code). Human-readable, git-friendly, no DB dependency.
5. **Scratchpad over message passing**: Workers communicate via filesystem, not message queues. Simpler for POC.
6. **Lazy loading via flags**: Python can't do compile-time DCE, but conditional imports achieve the same effect.

## Inspirations

- **Claude Code** (leak 2026-03-31): Dream system, Coordinator mode, Permission system, Tool registry with schema caching, Feature flags
- **Hermes Agent** (NousResearch): Model-agnostic philosophy, Skills system, Multi-platform, RL research integration, MIT license
- **OpenCode**: Skill system, Agent delegation patterns

## Future Considerations (Post-POC)

- Bridge system for IDE integration (VS Code, JetBrains)
- ML-specific tools (MLflow, drift detection, experiment tracking)
- Voice input
- Multi-platform messaging (Telegram, Discord)
- Rust rewrite of hot paths
- RL training integration (trajectory compression)
- Skills system with auto-creation (Hermes pattern)
