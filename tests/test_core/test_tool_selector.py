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
        available = [
            "bash",
            "file_read",
            "file_edit",
            "file_write",
            "grep",
            "glob",
            "web_fetch",
            "web_search",
            "model_inspector",
            "tensor_inspector",
        ]
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
        available = [
            "bash",
            "file_read",
            "file_edit",
            "file_write",
            "grep",
            "glob",
            "web_fetch",
            "model_inspector",
        ]
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
        available = [
            "bash",
            "file_read",
            "model_inspector",
            "tensor_inspector",
            "dataset_inspector",
            "model_eval",
            "model_card",
        ]
        selected = selector.select("Inspect the model weights and check tensor shapes", available)
        assert "model_inspector" in selected or "tensor_inspector" in selected

    def test_max_tools_respected(self):
        """Never returns more than max_tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=3)
        available = [
            "bash",
            "file_read",
            "file_edit",
            "file_write",
            "grep",
            "glob",
            "web_fetch",
            "web_search",
            "model_inspector",
        ]
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
        assert "bash" in selected
        assert "file_read" in selected
        assert len(selected) <= 8

    def test_disabled_returns_all(self):
        """When enabled=False, returns all available tools."""
        selector = ToolSelector(always_include=["bash"], max_tools=4, enabled=False)
        available = [
            "bash",
            "file_read",
            "file_edit",
            "file_write",
            "grep",
            "glob",
            "web_fetch",
            "web_search",
            "model_inspector",
        ]
        selected = selector.select("anything", available)
        assert selected == available

    def test_history_context_contributes(self):
        """Recent assistant messages expand tool selection."""
        selector = ToolSelector(always_include=["bash"], max_tools=8)
        available = ["bash", "file_read", "grep", "glob", "web_fetch", "web_search"]
        selected = selector.select(
            "continue with the next step",
            available,
            recent_context="I fetched the documentation from the web",
        )
        assert "web_fetch" in selected or "web_search" in selected


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

        # Registry with many tools
        mock_registry = MagicMock()
        all_schemas = [
            {"name": "bash", "description": "run bash"},
            {"name": "file_read", "description": "read file"},
            {"name": "file_edit", "description": "edit file"},
            {"name": "file_write", "description": "write file"},
            {"name": "grep", "description": "grep"},
            {"name": "glob", "description": "glob"},
            {"name": "web_fetch", "description": "fetch url"},
            {"name": "web_search", "description": "search web"},
            {"name": "model_inspector", "description": "inspect model"},
        ]
        mock_registry.get_schemas.return_value = all_schemas

        selector = ToolSelector(
            always_include=["bash", "file_read"],
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

        await loop.run("hello world")

        # Verify tools sent to LLM were filtered
        call_args = mock_llm.complete.call_args
        tools_sent = (
            call_args.kwargs.get("tools") if call_args.kwargs else call_args[1].get("tools")
        )
        assert tools_sent is not None
        assert len(tools_sent) <= 4
        # Core tools present
        tool_names = [t["name"] for t in tools_sent]
        assert "bash" in tool_names
        assert "file_read" in tool_names
