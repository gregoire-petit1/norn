# Norn CLI UX — Streaming, Tool Display, Multiline Input

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform Norn's CLI from batch-response to streaming, with rich tool progress and multiline input support.

**Architecture:** Three layers modified bottom-up: LLM provider (streaming with tool call accumulation), agent loop (new `run_stream()` yielding events), CLI (Rich Live rendering with progressive markdown). Each layer is independently testable.

**Tech Stack:** litellm (async streaming), Rich (Live, Spinner, Console), prompt_toolkit (multiline input)

---

## Task 1: Enhance StreamChunk model + add AgentEvent

**Files:**
- Modify: `src/norn/core/models.py`
- Test: `tests/test_core/test_models.py`

**Step 1: Write failing tests**

```python
# In tests/test_core/test_models.py — append these tests

from norn.core.models import AgentEvent, EventType, StreamChunk, TokenUsage


def test_stream_chunk_has_usage_field():
    chunk = StreamChunk(content="hi", done=True, usage=TokenUsage(prompt_tokens=10))
    assert chunk.usage.prompt_tokens == 10


def test_agent_event_text_type():
    event = AgentEvent(type=EventType.TEXT_DELTA, content="hello")
    assert event.type == EventType.TEXT_DELTA
    assert event.content == "hello"


def test_agent_event_tool_start_type():
    event = AgentEvent(type=EventType.TOOL_START, tool_name="bash", tool_args="ls")
    assert event.tool_name == "bash"


def test_agent_event_tool_end_type():
    event = AgentEvent(
        type=EventType.TOOL_END, tool_name="bash", tool_args="ls",
        duration_ms=150, success=True,
    )
    assert event.duration_ms == 150


def test_agent_event_done_type():
    event = AgentEvent(
        type=EventType.DONE,
        usage=TokenUsage(prompt_tokens=100, completion_tokens=50, total_tokens=150),
        latency_ms=2000,
    )
    assert event.usage.total_tokens == 150
```

**Step 2: Run tests — expect FAIL** (AgentEvent/EventType don't exist)

```bash
uv run pytest tests/test_core/test_models.py -v -k "agent_event or stream_chunk_has_usage"
```

**Step 3: Implement models**

Add to `src/norn/core/models.py`:

```python
class EventType(StrEnum):
    """Types of events yielded by the streaming agent loop."""
    TEXT_DELTA = "text_delta"      # Partial text from LLM
    TOOL_START = "tool_start"     # Tool execution beginning
    TOOL_END = "tool_end"         # Tool execution complete
    DONE = "done"                 # Turn complete


class AgentEvent(BaseModel):
    """Event yielded by AgentLoop.run_stream()."""
    type: EventType
    content: str | None = None          # For TEXT_DELTA
    tool_name: str | None = None        # For TOOL_START/TOOL_END
    tool_args: str | None = None        # For TOOL_START/TOOL_END
    tool_result: str | None = None      # For TOOL_END
    duration_ms: int | None = None      # For TOOL_END, DONE
    success: bool | None = None         # For TOOL_END
    usage: TokenUsage | None = None     # For DONE
    latency_ms: int | None = None       # For DONE
    model: str | None = None            # For DONE
```

Also add `usage` field to `StreamChunk`:
```python
class StreamChunk(BaseModel):
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    done: bool = False
    usage: TokenUsage | None = None
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(models): add AgentEvent, EventType for streaming agent loop"
```

---

## Task 2: Fix LiteLLMProvider.stream() — tool call accumulation + observability

**Files:**
- Modify: `src/norn/core/llm.py`
- Test: `tests/test_core/test_llm.py`

**Step 1: Write failing tests**

```python
# Append to tests/test_core/test_llm.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from norn.core.llm import LiteLLMProvider
from norn.core.models import Message, Role, StreamChunk


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Build a mock litellm streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


async def _mock_stream_response(chunks):
    """Create an async iterable from a list of mock chunks."""
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_yields_text_chunks():
    chunks = [
        _make_stream_chunk(content="Hello"),
        _make_stream_chunk(content=" world"),
        _make_stream_chunk(content=None, finish_reason="stop"),
    ]

    async def mock_completion(**kwargs):
        return _mock_stream_response(chunks)

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="hi")]

    collected = []
    async for chunk in provider.stream(messages):
        collected.append(chunk)

    assert any(c.content == "Hello" for c in collected)
    assert any(c.content == " world" for c in collected)
    assert collected[-1].done is True


@pytest.mark.asyncio
async def test_stream_accumulates_tool_calls():
    """Tool call fragments across chunks should be assembled into complete ToolCalls."""
    tc_frag_1 = MagicMock()
    tc_frag_1.index = 0
    tc_frag_1.id = "call_abc"
    tc_frag_1.function = MagicMock()
    tc_frag_1.function.name = "bash"
    tc_frag_1.function.arguments = '{"comm'

    tc_frag_2 = MagicMock()
    tc_frag_2.index = 0
    tc_frag_2.id = None
    tc_frag_2.function = MagicMock()
    tc_frag_2.function.name = None
    tc_frag_2.function.arguments = 'and": "ls"}'

    chunks = [
        _make_stream_chunk(tool_calls=[tc_frag_1]),
        _make_stream_chunk(tool_calls=[tc_frag_2]),
        _make_stream_chunk(finish_reason="tool_calls"),
    ]

    async def mock_completion(**kwargs):
        return _mock_stream_response(chunks)

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="run ls")]

    collected = []
    async for chunk in provider.stream(messages):
        collected.append(chunk)

    final = collected[-1]
    assert final.done is True
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "bash"
    assert final.tool_calls[0].arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_stream_uses_completion_fn():
    """stream() should use self._completion_fn, not hardcoded litellm."""
    called_with = {}

    async def mock_completion(**kwargs):
        called_with.update(kwargs)
        return _mock_stream_response([
            _make_stream_chunk(content="ok", finish_reason="stop"),
        ])

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="hi")]

    async for _ in provider.stream(messages):
        pass

    assert called_with["stream"] is True
    assert called_with["model"] == "test/model"
```

**Step 2: Run tests — expect FAIL**

**Step 3: Rewrite `LiteLLMProvider.stream()`**

Key changes:
- Use `self._completion_fn` instead of hardcoded `litellm.acompletion`
- Apply prompt-cache markers
- Accumulate tool call fragments by index
- On `done`, yield final chunk with assembled tool_calls + usage
- Add observability

```python
async def stream(
    self,
    messages: list[Message],
    tools: list[dict] | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> AsyncIterator[StreamChunk]:
    if not messages:
        raise ValueError("messages cannot be empty")

    kwargs: dict = {
        "model": self.model,
        "messages": _messages_to_dicts(messages),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if self.api_base:
        kwargs["api_base"] = self.api_base

    tool_schemas = build_tool_schemas(tools or [])
    if tool_schemas:
        kwargs["tools"] = tool_schemas

    if self._prompt_cache:
        marked_messages, marked_tools = _apply_cache_markers(
            kwargs["messages"], kwargs.get("tools"), enabled=True,
        )
        kwargs["messages"] = marked_messages
        if marked_tools is not None:
            kwargs["tools"] = marked_tools

    completion = self._completion_fn or litellm.acompletion
    start = __import__("time").monotonic()
    response = await completion(**kwargs)

    # Accumulator for fragmented tool calls
    tc_accum: dict[int, dict] = {}  # index -> {id, name, arguments_str}
    final_usage = None

    async for chunk in response:
        choice = chunk.choices[0] if chunk.choices else None
        if choice is None:
            continue

        delta = choice.delta

        # Accumulate tool call fragments
        if hasattr(delta, "tool_calls") and delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in tc_accum:
                    tc_accum[idx] = {"id": "", "name": "", "arguments": ""}
                if tc_delta.id:
                    tc_accum[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        tc_accum[idx]["name"] = tc_delta.function.name
                    if tc_delta.function.arguments:
                        tc_accum[idx]["arguments"] += tc_delta.function.arguments

        # Track usage from final chunk
        if hasattr(chunk, "usage") and chunk.usage:
            final_usage = chunk.usage

        content = delta.content if hasattr(delta, "content") else None
        done = choice.finish_reason is not None

        if done:
            # Assemble accumulated tool calls
            assembled_calls = None
            if tc_accum:
                assembled_calls = []
                for idx in sorted(tc_accum):
                    tc = tc_accum[idx]
                    try:
                        args = json.loads(tc["arguments"]) if tc["arguments"] else {}
                    except (json.JSONDecodeError, ValueError):
                        args = {}
                    assembled_calls.append(
                        ToolCall(id=tc["id"], name=tc["name"], arguments=args)
                    )

            usage = None
            if final_usage:
                usage = TokenUsage(
                    prompt_tokens=getattr(final_usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(final_usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(final_usage, "total_tokens", 0) or 0,
                )

            yield StreamChunk(
                content=content,
                tool_calls=assembled_calls,
                done=True,
                usage=usage,
            )
        else:
            yield StreamChunk(content=content, done=False)
```

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(llm): fix stream() — tool call accumulation, cache markers, completion_fn"
```

---

## Task 3: Add AgentLoop.run_stream() yielding AgentEvents

**Files:**
- Modify: `src/norn/core/agent.py`
- Test: `tests/test_core/test_agent.py`

**Step 1: Write failing tests**

Test `run_stream()` as an async generator yielding `AgentEvent` objects:
- TEXT_DELTA events during streaming
- TOOL_START / TOOL_END events during tool execution
- DONE event at the end
- Tool call fallback extraction still works
- Multi-round tool loops work (stream → tool → stream → done)

**Step 2: Run tests — expect FAIL**

**Step 3: Implement `run_stream()`**

Key design:
- `run_stream(user_input) -> AsyncIterator[AgentEvent]`
- Calls `self.llm.stream()` instead of `self.llm.complete()`
- Yields `TEXT_DELTA` for each content chunk
- When stream finishes with tool_calls: yield `TOOL_START`, execute, yield `TOOL_END`, re-loop
- When stream finishes without tool_calls: yield `DONE`
- Tool call text fallback: if stream finishes with content but no tool_calls, try extractor
- Truncation applied to tool results (same as `_run_impl`)
- History management same as `_run_impl`

**Step 4: Run tests — expect PASS**

**Step 5: Commit**

```bash
git commit -m "feat(agent): add run_stream() yielding AgentEvent for real-time CLI rendering"
```

---

## Task 4: Rich streaming CLI renderer

**Files:**
- Modify: `src/norn/cli/main.py`
- Create: `src/norn/cli/renderer.py`
- Test: manual (Rich output is visual)

**Step 1: Create renderer module**

`src/norn/cli/renderer.py` — encapsulates Rich rendering logic:

```python
class StreamRenderer:
    """Renders AgentEvents to the terminal using Rich."""

    def __init__(self, console: Console):
        self.console = console
        self._text_buffer = ""

    async def render(self, events: AsyncIterator[AgentEvent]) -> None:
        """Consume an event stream and render to console."""
        async for event in events:
            match event.type:
                case EventType.TEXT_DELTA:
                    # Print text incrementally (raw, not markdown)
                    if event.content:
                        self._text_buffer += event.content
                        self.console.print(event.content, end="")
                case EventType.TOOL_START:
                    # Flush text buffer as markdown before tool output
                    self._flush_markdown()
                    self.console.print(
                        f"  [dim][tool] {event.tool_name}: {event.tool_args}...[/dim]",
                        end="",
                    )
                case EventType.TOOL_END:
                    status = "" if event.success else " [red]FAIL[/red]"
                    self.console.print(
                        f" ({event.duration_ms / 1000:.1f}s){status}",
                        style="dim",
                    )
                case EventType.DONE:
                    self._flush_markdown()
                    self._print_metrics(event)

    def _flush_markdown(self):
        """Render accumulated text as markdown."""
        if self._text_buffer.strip():
            # Re-render as markdown for proper formatting
            self.console.print()  # newline after raw streaming
            self.console.print(Markdown(self._text_buffer))
            self._text_buffer = ""

    def _print_metrics(self, event: AgentEvent):
        """Print metrics line."""
        parts = []
        if event.latency_ms:
            parts.append(f"{event.latency_ms / 1000:.1f}s")
        if event.usage:
            parts.append(f"{event.usage.prompt_tokens}→{event.usage.completion_tokens} tokens")
        if event.model:
            parts.append(event.model)
        if parts:
            self.console.print(f"  ⏱ {'│'.join(parts)}", style="dim")
```

**Step 2: Update CLI chat/run commands**

Replace:
```python
response = await agent.run(user_input)
if response.content:
    console.print(Markdown(response.content))
_print_metrics(response)
```

With:
```python
renderer = StreamRenderer(console)
await renderer.render(agent.run_stream(user_input))
```

Keep `agent.run()` for backward compat (bench, coordinator, etc.)

**Step 3: Commit**

```bash
git commit -m "feat(cli): stream responses with Rich progressive rendering"
```

---

## Task 5: Better tool call display

**Files:**
- Modify: `src/norn/cli/renderer.py`

**Step 1: Enhance tool display**

- Color by risk level: read=green, write=yellow, bash=red
- Spinner during tool execution (Rich Status)
- Collapsible tool output for long results

```python
# Tool risk colors
_TOOL_COLORS = {
    "file_read": "green", "glob": "green", "grep": "green",
    "file_write": "yellow", "file_edit": "yellow",
    "bash": "red",
}

case EventType.TOOL_START:
    self._flush_markdown()
    color = _TOOL_COLORS.get(event.tool_name, "blue")
    self._spinner = self.console.status(
        f"[{color}]{event.tool_name}[/{color}]: {event.tool_args}",
        spinner="dots",
    )
    self._spinner.start()

case EventType.TOOL_END:
    if self._spinner:
        self._spinner.stop()
    status = "✓" if event.success else "[red]✗[/red]"
    color = _TOOL_COLORS.get(event.tool_name, "blue")
    self.console.print(
        f"  {status} [{color}]{event.tool_name}[/{color}]: "
        f"{event.tool_args} ({event.duration_ms / 1000:.1f}s)",
        style="dim",
    )
```

**Step 2: Commit**

```bash
git commit -m "feat(cli): colored tool progress with spinners"
```

---

## Task 6: Multiline input with prompt_toolkit

**Files:**
- Modify: `src/norn/cli/main.py`
- Modify: `pyproject.toml` (add `prompt_toolkit` dependency)

**Step 1: Add dependency**

```bash
uv add prompt_toolkit
```

**Step 2: Replace Rich Prompt.ask with prompt_toolkit**

```python
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

def _build_prompt_session() -> PromptSession:
    """Build a prompt_toolkit session with multiline support.

    - Enter sends the message
    - Alt+Enter / Esc+Enter inserts a newline
    - Ctrl+D exits
    """
    bindings = KeyBindings()

    @bindings.add("escape", "enter")
    def _(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(
        message="> ",
        multiline=False,  # Enter sends by default
        key_bindings=bindings,
    )
```

Update `_chat_loop()`:
```python
session = _build_prompt_session()
while True:
    try:
        user_input = await asyncio.get_event_loop().run_in_executor(
            None, lambda: session.prompt()
        )
    except (EOFError, KeyboardInterrupt):
        console.print("\nGoodbye.")
        break
```

**Step 3: Commit**

```bash
git commit -m "feat(cli): multiline input with prompt_toolkit (Alt+Enter for newlines)"
```

---

## Task 7: Integration test + final polish

**Files:**
- Run full test suite
- Manual smoke test of streaming + tools + multiline

**Step 1:** `uv run pytest --tb=short -q` — all green

**Step 2:** Manual test: `norn chat` → send a prompt → verify streaming output, tool spinners, multiline

**Step 3:** Final commit if any fixes needed

---

## Execution Order

| Task | Depends On | Est. Time |
|------|-----------|-----------|
| 1. Models | — | 10 min |
| 2. LLM stream fix | 1 | 20 min |
| 3. Agent run_stream | 1, 2 | 30 min |
| 4. CLI renderer | 1, 3 | 20 min |
| 5. Tool display | 4 | 15 min |
| 6. Multiline input | — | 15 min |
| 7. Integration | all | 10 min |
| **Total** | | **~2h** |
