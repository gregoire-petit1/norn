"""Tests for sliding window context manager."""

from __future__ import annotations

import pytest

from norn.core.context import ContextManager
from norn.core.models import Message, Role


def _make_history(n_turns: int, chars_per_msg: int = 100) -> list[Message]:
    """Create a fake conversation history with n user+assistant turns."""
    history: list[Message] = []
    for i in range(n_turns):
        history.append(Message(role=Role.USER, content=f"Question {i}: {'x' * chars_per_msg}"))
        history.append(Message(role=Role.ASSISTANT, content=f"Answer {i}: {'y' * chars_per_msg}"))
    return history


class TestContextManager:
    """Test ContextManager sliding window behavior."""

    @pytest.mark.asyncio
    async def test_short_history_unchanged(self):
        """History under budget is returned verbatim."""
        cm = ContextManager(max_history_tokens=50000, recent_turns_keep=6)
        history = _make_history(3)  # 6 messages, well under budget
        messages = await cm.build_messages("System prompt", history)
        # system + all history messages
        assert len(messages) == 1 + len(history)
        assert messages[0].role == Role.SYSTEM

    @pytest.mark.asyncio
    async def test_long_history_triggers_window(self):
        """History over budget gets windowed — fewer messages returned."""
        # Very small budget to force windowing
        cm = ContextManager(
            max_history_tokens=500,  # tiny budget
            recent_turns_keep=2,
        )
        history = _make_history(20, chars_per_msg=200)  # 40 messages, way over budget
        messages = await cm.build_messages("System prompt", history)
        # Should have far fewer messages than 1 + 40
        assert len(messages) < 10
        # Should contain a summary somewhere
        all_content = " ".join(m.content or "" for m in messages)
        assert "User:" in all_content or "Summary" in all_content or "Question" in all_content

    @pytest.mark.asyncio
    async def test_recent_turns_always_kept(self):
        """The last N turns are always preserved verbatim."""
        cm = ContextManager(max_history_tokens=100, recent_turns_keep=3)
        history = _make_history(10, chars_per_msg=50)
        messages = await cm.build_messages("System", history)
        # Last 3 turns = last 6 messages from history
        last_6 = history[-6:]
        # They should appear verbatim at the end of messages
        for orig in last_6:
            assert any(m.content == orig.content for m in messages)

    @pytest.mark.asyncio
    async def test_summary_cached(self):
        """Summary is cached and not regenerated every call with same history."""
        cm = ContextManager(max_history_tokens=200, recent_turns_keep=2)
        history = _make_history(10, chars_per_msg=100)
        # First call generates summary
        await cm.build_messages("System", history)
        first_summary = cm._cached_summary
        assert first_summary is not None
        # Second call with same history length uses cache
        await cm.build_messages("System", history)
        assert cm._cached_summary is first_summary  # same object

    @pytest.mark.asyncio
    async def test_disabled_returns_full_history(self):
        """When enabled=False, returns full history unmodified."""
        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2, enabled=False)
        history = _make_history(20, chars_per_msg=200)
        messages = await cm.build_messages("System", history)
        assert len(messages) == 1 + len(history)  # system + all

    def test_token_estimate(self):
        """Token estimation works with chars/4 heuristic."""
        cm = ContextManager(max_history_tokens=1000, recent_turns_keep=2)
        msgs = _make_history(5, chars_per_msg=100)
        tokens = cm._estimate_tokens(msgs)
        # ~110 chars per msg * 10 msgs / 4 ≈ 275 tokens
        assert 200 < tokens < 400

    @pytest.mark.asyncio
    async def test_empty_history(self):
        """Empty history returns just system message."""
        cm = ContextManager(max_history_tokens=1000, recent_turns_keep=6)
        messages = await cm.build_messages("System prompt", [])
        assert len(messages) == 1
        assert messages[0].content == "System prompt"


class TestAgentIntegration:
    """Test context manager integration with AgentLoop."""

    @pytest.mark.asyncio
    async def test_agent_uses_context_manager(self):
        """AgentLoop with context_manager uses windowed messages."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop
        from norn.core.context import ContextManager
        from norn.core.models import LLMResponse, Message, Role

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(content="ok"))
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2, enabled=True)

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            context_manager=cm,
            env_bootstrap=False,
        )

        # Simulate many prior turns
        for i in range(20):
            loop.history.append(Message(role=Role.USER, content=f"Q{i}: {'x' * 200}"))
            loop.history.append(Message(role=Role.ASSISTANT, content=f"A{i}: {'y' * 200}"))

        # Next turn should use windowed context
        await loop.run("final question")

        # Verify the messages sent to LLM are windowed (not 42+ messages)
        call_args = mock_llm.complete.call_args
        messages_sent = call_args.kwargs.get("messages") if call_args.kwargs else call_args[0][0]
        # Should be system + summary + recent (2 turns × 2 = 4) + the new user msg
        assert len(messages_sent) < 12

    @pytest.mark.asyncio
    async def test_agent_without_context_manager_sends_full(self):
        """AgentLoop without context_manager sends full history."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop
        from norn.core.models import LLMResponse, Message, Role

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(content="ok"))
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            context_manager=None,
            env_bootstrap=False,
        )

        # Add some history
        for i in range(5):
            loop.history.append(Message(role=Role.USER, content=f"Q{i}"))
            loop.history.append(Message(role=Role.ASSISTANT, content=f"A{i}"))

        await loop.run("final")

        call_args = mock_llm.complete.call_args
        messages_sent = call_args.kwargs.get("messages") if call_args.kwargs else call_args[0][0]
        # system + 10 history + 1 new user = 12
        assert len(messages_sent) == 12


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
                tool_calls=[
                    ToolCall(id="t1", name="bash", arguments={"command": "find . -name '*.py'"})
                ],
            ),
            Message(
                role=Role.TOOL,
                content="src/main.py\nsrc/utils.py\nsrc/config.py\n" * 50,
                tool_call_id="t1",
            ),
            Message(role=Role.ASSISTANT, content="Found 150 Python files in src/"),
        ]
        summary = cm._extractive_summary(messages)
        # Should mention the tool was used and key outcome
        assert "bash" in summary.lower() or "find" in summary.lower()
        # Should NOT contain the full long output
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
            Message(
                role=Role.ASSISTANT,
                content="The project follows a layered architecture with core, tools, and CLI modules.",
            ),
        ]
        summary = cm._extractive_summary(messages)
        assert "architecture" in summary.lower() or "User:" in summary
