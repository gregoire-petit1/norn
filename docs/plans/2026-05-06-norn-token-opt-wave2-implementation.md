# Token Optimization Wave 2 Completion — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement dynamic tool selection (send only relevant tool schemas per turn) and tool call collapse (compress evicted tool_call+result pairs into brief summaries).

**Architecture:** W2.3 adds a `ToolSelector` that picks 3-8 tools per turn based on keyword heuristics. W2.4 enhances `ContextManager._extractive_summary()` to collapse tool messages intelligently. Both are additive, config-gated.

**Tech Stack:** Python 3.11+, Pydantic, pytest, existing `ToolRegistry.scoped()` method.

**Design reference:** `docs/plans/2026-04-22-norn-token-optimization-design.md` §4.5 and §4.6.

---

## Task 1: Dynamic Tool Selection — Core Module

**Files:**
- Create: `src/norn/core/tool_selector.py`
- Test: `tests/test_core/test_tool_selector.py`

**Step 1: Write the failing tests**

```python
# tests/test_core/test_tool_selector.py
"""Tests for dynamic tool selection (W2.3)."""

from __future__ import annotations

import pytest

from norn.core.tool_selector import ToolSelector


class TestToolSelector:
    """Test keyword-based tool selection."""

    def test_always_includes_core_tools(self):
        """Core tools are always included regardless of message content."""
        selector = ToolSelector(
            always_include=["bash", "file_read", "file_edit", "file_write"],
            max_tools=8,
        )
        available = ["bash", "file_read", "file_edit", "file_write", "grep", "glob",
                     "web_fetch", "web_search", "model_inspector", "tensor_inspector"]
        selected = selector.select("hello", available)
        assert "bash" in selected
        assert "file_read" in selected
        assert "file_edit" in selected
        assert "file_write" in selected

    def test_file_keywords_select_file_tools(self):
        """File-related keywords select grep and glob."""
        selector = ToolSelector(
            always_include=["bash"],
            max_tools=8,
        )
        available = ["bash", "file_read", "file_edit", "file_write", "grep", "glob",
                     "web_fetch", "model_inspector"]
        selected = selector.select("Find all Python files in the project", available)
        assert "grep" in selected or "glob" in selected

    def test_web_keywords_select_web_tools(self):
        """Web-related keywords select web tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=8)
        available = ["bash", "file_read", "grep", "web_fetch", "web_search"]
        selected = selector.select("Search the web for PyTorch documentation", available)
        assert "web_fetch" in selected
        assert "web_search" in selected

    def test_ml_keywords_select_ml_tools(self):
        """ML-related keywords select ML tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=8)
        available = ["bash", "file_read", "model_inspector", "tensor_inspector",
                     "dataset_inspector", "model_eval", "model_card"]
        selected = selector.select("Inspect the model weights and check tensor shapes", available)
        assert "model_inspector" in selected or "tensor_inspector" in selected

    def test_max_tools_respected(self):
        """Never returns more than max_tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=3)
        available = ["bash", "file_read", "file_edit", "file_write", "grep", "glob",
                     "web_fetch", "web_search", "model_inspector"]
        selected = selector.select("read and edit files then search for code", available)
        assert len(selected) <= 3

    def test_no_match_returns_all_up_to_max(self):
        """When no keywords match, return core + fill up to max."""
        selector = ToolSelector(
            always_include=["bash", "file_read"],
            max_tools=8,
        )
        available = ["bash", "file_read", "file_edit", "file_write", "grep", "glob"]
        selected = selector.select("do something", available)
        # Should include core + others up to max
        assert "bash" in selected
        assert "file_read" in selected
        assert len(selected) <= 8

    def test_disabled_returns_all(self):
        """When enabled=False, returns all available tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=4, enabled=False)
        available = ["bash", "file_read", "file_edit", "file_write", "grep", "glob",
                     "web_fetch", "web_search", "model_inspector"]
        selected = selector.select("anything", available)
        assert selected == available

    def test_history_context_contributes(self):
        """Recent assistant messages expand tool selection."""
        selector = ToolSelector(always_include=["bash"], max_tools=8)
        available = ["bash", "file_read", "grep", "glob", "web_fetch", "web_search"]
        # User asks a question, but recent history mentions web
        selected = selector.select(
            "continue with the next step",
            available,
            recent_context="I fetched the documentation from the web",
        )
        assert "web_fetch" in selected or "web_search" in selected
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_tool_selector.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Write the implementation**

```python
# src/norn/core/tool_selector.py
"""Dynamic tool selection via keyword heuristics (W2.3).

Selects a subset of available tools per turn based on message content,
reducing tool schema tokens sent to the LLM by 30-80%.
"""

from __future__ import annotations

import re

# Keyword -> tool groups mapping
_TOOL_GROUPS: dict[str, list[str]] = {
    "file_ops": ["file_read", "file_edit", "file_write", "grep", "glob"],
    "search": ["grep", "glob"],
    "web": ["web_fetch", "web_search"],
    "ml": ["model_inspector", "tensor_inspector", "dataset_inspector", "model_eval", "model_card"],
}

# Keywords that trigger each tool group
_KEYWORD_PATTERNS: dict[str, re.Pattern[str]] = {
    "file_ops": re.compile(
        r"\b(?:file|read|write|edit|create|delete|rename|path|directory|folder)\b", re.IGNORECASE
    ),
    "search": re.compile(
        r"\b(?:find|search|grep|glob|pattern|locate|look for|where is)\b", re.IGNORECASE
    ),
    "web": re.compile(
        r"\b(?:web|url|http|fetch|download|browse|internet|online|documentation)\b", re.IGNORECASE
    ),
    "ml": re.compile(
        r"\b(?:model|tensor|dataset|weight|checkpoint|inference|train|eval|metric|inspect)\b",
        re.IGNORECASE,
    ),
}


class ToolSelector:
    """Select relevant tools per turn using keyword heuristics.

    Args:
        always_include: Tool names always included (core set).
        max_tools: Maximum tools to return per turn.
        enabled: If False, returns all available tools (bypass).
    """

    def __init__(
        self,
        always_include: list[str] | None = None,
        max_tools: int = 8,
        enabled: bool = True,
    ) -> None:
        self.always_include = always_include or ["bash", "file_read", "file_edit", "file_write"]
        self.max_tools = max_tools
        self.enabled = enabled

    def select(
        self,
        user_message: str,
        available_tools: list[str],
        *,
        recent_context: str | None = None,
    ) -> list[str]:
        """Select tools relevant to the current message.

        Args:
            user_message: The current user message.
            available_tools: All available tool names.
            recent_context: Optional text from recent assistant messages for broader matching.

        Returns:
            List of selected tool names (subset of available_tools).
        """
        if not self.enabled:
            return available_tools

        # Combine user message with recent context for matching
        search_text = user_message
        if recent_context:
            search_text = f"{user_message} {recent_context}"

        # Start with core tools
        selected: set[str] = set()
        for name in self.always_include:
            if name in available_tools:
                selected.add(name)

        # Match keyword patterns to tool groups
        for group_name, pattern in _KEYWORD_PATTERNS.items():
            if pattern.search(search_text):
                for tool_name in _TOOL_GROUPS[group_name]:
                    if tool_name in available_tools:
                        selected.add(tool_name)

        # If nothing matched beyond core, add all available up to max
        if len(selected) <= len(self.always_include):
            for name in available_tools:
                selected.add(name)
                if len(selected) >= self.max_tools:
                    break

        # Cap at max_tools (preserve core tools priority)
        if len(selected) > self.max_tools:
            core = {n for n in self.always_include if n in available_tools}
            non_core = [n for n in selected if n not in core]
            selected = core | set(non_core[: self.max_tools - len(core)])

        return [name for name in available_tools if name in selected]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_tool_selector.py -v`
Expected: All 8 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/tool_selector.py tests/test_core/test_tool_selector.py
git commit -m "feat(tool-selector): add keyword-based dynamic tool selection

ToolSelector picks 3-8 relevant tools per turn based on message keywords.
Core tools (bash, file_read, file_edit, file_write) always included.
Groups: file_ops, search, web, ml. Reduces tool schema tokens by 30-80%."
```

---

## Task 2: Dynamic Tool Selection — Wire into AgentLoop

**Files:**
- Modify: `src/norn/core/agent.py` (add ToolSelector, use in both run methods)
- Modify: `src/norn/core/config.py` (add dynamic_tools fields to ContextConfig)
- Test: `tests/test_core/test_tool_selector.py` (add integration test)

**Step 1: Write the failing test**

```python
# Add to tests/test_core/test_tool_selector.py

class TestAgentIntegration:
    """Test tool selector integration with AgentLoop."""

    @pytest.mark.asyncio
    async def test_agent_sends_subset_of_tools(self):
        """AgentLoop with tool_selector sends only selected tool schemas."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop
        from norn.core.models import LLMResponse
        from norn.core.tool_selector import ToolSelector

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(content="ok"))

        # Registry with 8 tools
        mock_registry = MagicMock()
        all_schemas = [{"name": f"tool_{i}"} for i in range(8)]
        mock_registry.get_schemas.return_value = all_schemas
        mock_registry.list_tools.return_value = [
            MagicMock(name=f"tool_{i}") for i in range(8)
        ]

        selector = ToolSelector(
            always_include=["tool_0", "tool_1"],
            max_tools=4,
            enabled=True,
        )

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            tool_selector=selector,
            env_bootstrap=False,
        )

        await loop.run("do something simple")

        # Verify tools sent to LLM were filtered
        call_args = mock_llm.complete.call_args
        tools_sent = call_args.kwargs.get("tools")
        # Should be fewer than all 8 schemas
        assert tools_sent is None or len(tools_sent) <= 4
```

**Step 2: Run test → fail**

**Step 3: Modify agent.py and config.py**

Add to `ContextConfig`:
```python
class ContextConfig(BaseModel):
    sliding_window: bool = False
    max_history_tokens: int = 8000
    recent_turns_keep: int = 6
    summary_max_tokens: int = 300
    summary_model: str | None = None
    # Dynamic tool selection (W2.3)
    dynamic_tools: bool = False  # opt-in
    always_include_tools: list[str] = Field(
        default_factory=lambda: ["bash", "file_read", "file_edit", "file_write"]
    )
    max_tools_per_turn: int = 8
```

Add `tool_selector: ToolSelector | None = None` to AgentLoop.__init__().

In both `_run_impl` and `run_stream`, before calling `self.llm.complete()/stream()`:
- Get available tool names from registry
- If tool_selector exists, select subset
- Filter schemas to only selected tools

**Step 4: Run tests, full suite**

**Step 5: Commit**

```bash
git add src/norn/core/agent.py src/norn/core/config.py tests/test_core/test_tool_selector.py
git commit -m "feat(tool-selector): wire into AgentLoop for dynamic tool filtering

AgentLoop accepts optional tool_selector. When present, only relevant
tool schemas are sent to the LLM each turn. Opt-in via config:
context.dynamic_tools=true."
```

---

## Task 3: Tool Call Collapse — Enhance Extractive Summary

**Files:**
- Modify: `src/norn/core/context.py` (enhance `_extractive_summary`)
- Test: `tests/test_core/test_context_manager.py` (add collapse-specific tests)

**Step 1: Write the failing tests**

```python
# Add to tests/test_core/test_context_manager.py

class TestToolCallCollapse:
    """Test tool call collapse during summarization."""

    def test_tool_messages_collapsed(self):
        """Tool call + result pairs are collapsed into brief summaries."""
        from norn.core.models import ToolCall

        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2)

        messages = [
            Message(role=Role.USER, content="Find all Python files"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="t1", name="bash", arguments={"command": "find . -name '*.py'"})],
            ),
            Message(role=Role.TOOL, content="src/main.py\nsrc/utils.py\nsrc/config.py\n" * 50, tool_call_id="t1"),
            Message(role=Role.ASSISTANT, content="Found 150 Python files in src/"),
        ]
        summary = cm._extractive_summary(messages)
        # Should mention the tool was used and key outcome
        assert "bash" in summary.lower() or "find" in summary.lower()
        # Should NOT contain the full 150-line output
        assert len(summary) < 500

    def test_multiple_tool_calls_collapsed(self):
        """Multiple sequential tool calls are each collapsed."""
        from norn.core.models import ToolCall

        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2)

        messages = [
            Message(role=Role.USER, content="Read the config and run tests"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="t1", name="file_read", arguments={"path": "config.yaml"})],
            ),
            Message(role=Role.TOOL, content="key: value\n" * 100, tool_call_id="t1"),
            Message(
                role=Role.ASSISTANT,
                content=None,
                tool_calls=[ToolCall(id="t2", name="bash", arguments={"command": "pytest"})],
            ),
            Message(role=Role.TOOL, content="5 passed\n", tool_call_id="t2"),
            Message(role=Role.ASSISTANT, content="Config loaded and all 5 tests pass."),
        ]
        summary = cm._extractive_summary(messages)
        # Both tools mentioned
        assert "file_read" in summary.lower() or "config" in summary.lower()
        assert "bash" in summary.lower() or "pytest" in summary.lower() or "test" in summary.lower()

    def test_non_tool_messages_preserved(self):
        """Regular user/assistant messages still included in summary."""
        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2)

        messages = [
            Message(role=Role.USER, content="Explain the architecture of this project"),
            Message(role=Role.ASSISTANT, content="The project follows a layered architecture with core, tools, and CLI modules."),
        ]
        summary = cm._extractive_summary(messages)
        assert "architecture" in summary.lower() or "User:" in summary
```

**Step 2: Run tests → fail (current extractive_summary skips tool messages entirely)**

**Step 3: Enhance `_extractive_summary` in context.py**

Replace the current implementation with one that handles tool messages:

```python
def _extractive_summary(self, messages: list[Message]) -> str:
    """Extractive summary with tool call collapse.

    Collapses tool_call + tool_result pairs into brief one-liners.
    Preserves user questions and key assistant responses.
    """
    parts: list[str] = []
    i = 0
    while i < len(messages) and len(parts) < 12:
        msg = messages[i]

        if msg.role == Role.USER and msg.content:
            parts.append(f"- User: {msg.content[:100]}")
        elif msg.role == Role.ASSISTANT:
            if msg.tool_calls:
                # Collapse tool call: show tool name + brief args
                for tc in msg.tool_calls:
                    args_summary = self._summarize_tool_args(tc)
                    # Look ahead for the tool result
                    result_preview = ""
                    for j in range(i + 1, min(i + len(msg.tool_calls) + 1, len(messages))):
                        if messages[j].role == Role.TOOL and messages[j].tool_call_id == tc.id:
                            result_content = messages[j].content or ""
                            result_preview = result_content[:60].replace("\n", " ")
                            break
                    parts.append(f"- Tool[{tc.name}]: {args_summary} → {result_preview}")
            elif msg.content and len(msg.content) > 20:
                parts.append(f"- Assistant: {msg.content[:80]}")
        # Skip standalone TOOL messages (already handled via look-ahead)
        i += 1

    return "\n".join(parts) if parts else "No prior context."

@staticmethod
def _summarize_tool_args(tool_call: ToolCall) -> str:
    """Produce a brief summary of tool call arguments."""
    args = tool_call.arguments
    if "command" in args:
        cmd = str(args["command"])
        return cmd[:40] + ("..." if len(cmd) > 40 else "")
    if "path" in args:
        return str(args["path"])
    if "file_path" in args:
        return str(args["file_path"])
    if "pattern" in args:
        return str(args["pattern"])
    for v in args.values():
        if isinstance(v, str):
            return v[:40]
    return "..."
```

Note: Need to import `ToolCall` in the TYPE_CHECKING block of context.py.

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_context_manager.py -v`
Expected: All tests PASS (old + new)

**Step 5: Commit**

```bash
git add src/norn/core/context.py tests/test_core/test_context_manager.py
git commit -m "feat(context): add tool call collapse to extractive summary

When summarizing evicted messages, tool_call + tool_result pairs are
now collapsed into brief one-liners (tool name + args + result preview)
instead of being skipped entirely. Reduces summary token waste by 40-60%."
```

---

## Task 4: CLI Wiring + Config + Final Integration

**Files:**
- Modify: `src/norn/cli/main.py` (pass tool_selector to AgentLoop)
- Modify: `configs/default.yaml` (document dynamic_tools settings)
- Test: run full suite

**Step 1: Update configs/default.yaml**

Add to the `context:` section:
```yaml
context:
  # ... existing sliding window settings ...
  # Dynamic tool selection (W2.3)
  dynamic_tools: false           # Opt-in: send only relevant tool schemas per turn
  always_include_tools:          # Core tools always sent
    - bash
    - file_read
    - file_edit
    - file_write
  max_tools_per_turn: 8          # Max tools sent per turn
```

**Step 2: Wire in CLI**

In `_build_context_manager()` or near AgentLoop construction, add:
```python
from norn.core.tool_selector import ToolSelector

tool_selector = None
if config.context.dynamic_tools:
    tool_selector = ToolSelector(
        always_include=config.context.always_include_tools,
        max_tools=config.context.max_tools_per_turn,
        enabled=True,
    )
```

Pass `tool_selector=tool_selector` to AgentLoop.

**Step 3: Run full suite**

Run: `uv run pytest --tb=short -q`
Expected: All tests pass

**Step 4: Commit**

```bash
git add src/norn/cli/main.py configs/default.yaml
git commit -m "feat(token-opt): wire dynamic tool selection into CLI

ToolSelector configurable via context.dynamic_tools, always_include_tools,
max_tools_per_turn. Opt-in (false by default). Completes Wave 2."
```

---

## Summary

| Task | Feature | Tests Added | Commits |
|------|---------|-------------|---------|
| 1 | Tool Selector core module | ~8 | 1 |
| 2 | Wire into AgentLoop | ~1 | 1 |
| 3 | Tool Call Collapse | ~3 | 1 |
| 4 | CLI + Config | — | 1 |
| **Total** | **W2.3 + W2.4** | **~12** | **4** |

**Estimated effort:** 1 session
