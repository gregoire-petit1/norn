"""Tests for tool call text fallback extraction."""

from __future__ import annotations

import pytest

from norn.core.models import ToolCall
from norn.core.tool_call_extractor import extract_tool_calls_from_text


# --- Pattern 1: OpenAI-style "Tool Calls: [...]" ---


class TestOpenAIPattern:
    """Model emits tool calls in OpenAI format as text."""

    def test_single_tool_call(self):
        text = (
            "I will read the file now.\n\n"
            'Tool Calls: [ { "id": "call_abc123", "type": "function", '
            '"function": { "name": "file_read", "arguments": '
            '{ "path": "src/solution.py" } } } ]'
        )
        calls, cleaned = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "file_read"
        assert calls[0].arguments == {"path": "src/solution.py"}
        assert calls[0].id  # should have an id
        assert "Tool Calls:" not in cleaned
        assert "I will read the file now." in cleaned

    def test_multiple_tool_calls(self):
        text = (
            "Tool Calls: [ "
            '{ "id": "call_1", "type": "function", '
            '"function": { "name": "file_read", "arguments": { "path": "a.py" } } }, '
            '{ "id": "call_2", "type": "function", '
            '"function": { "name": "file_read", "arguments": { "path": "b.py" } } } ]'
        )
        calls, cleaned = extract_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0].name == "file_read"
        assert calls[1].arguments == {"path": "b.py"}

    def test_preserves_original_id(self):
        text = (
            'Tool Calls: [ { "id": "call_xyz", "type": "function", '
            '"function": { "name": "bash", "arguments": '
            '{ "command": "pytest" } } } ]'
        )
        calls, _ = extract_tool_calls_from_text(text)
        assert calls[0].id == "call_xyz"


# --- Pattern 2: Flat JSON {"name": ..., "arguments": ...} ---


class TestFlatPattern:
    """Model emits tool calls in simplified flat JSON format."""

    def test_flat_json_tool_call(self):
        text = (
            "Let me run the tests.\n\n"
            '{"name": "bash", "arguments": {"command": "python -m pytest tests/ -v"}}'
        )
        calls, cleaned = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "bash"
        assert calls[0].arguments == {"command": "python -m pytest tests/ -v"}
        assert "Let me run the tests." in cleaned

    def test_flat_json_with_complex_arguments(self):
        text = (
            '{"name": "file_edit", "arguments": '
            '{"path": "src/solution.py", '
            '"old_string": "start = page * per_page", '
            '"new_string": "start = (page - 1) * per_page"}}'
        )
        calls, _ = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "file_edit"
        assert calls[0].arguments["old_string"] == "start = page * per_page"


# --- No tool calls ---


class TestNoToolCalls:
    """Text that does not contain tool calls."""

    def test_plain_text(self):
        text = "The function looks correct. No changes needed."
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text

    def test_empty_string(self):
        calls, cleaned = extract_tool_calls_from_text("")
        assert calls == []
        assert cleaned == ""

    def test_json_that_is_not_tool_call(self):
        """JSON in text that isn't a tool call should not be extracted."""
        text = 'The config is: {"debug": true, "port": 8080}'
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text

    def test_code_block_with_json(self):
        """JSON inside markdown code blocks should not be extracted."""
        text = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text


# --- Robustness ---


class TestRobustness:
    """Edge cases and malformed input."""

    def test_malformed_json(self):
        text = 'Tool Calls: [ { "name": "bash", bad json here } ]'
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []
        assert cleaned == text

    def test_tool_call_missing_name(self):
        """JSON with arguments but no name is not a tool call."""
        text = '{"arguments": {"path": "foo.py"}}'
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []

    def test_tool_call_missing_arguments(self):
        """JSON with name but no arguments is not a tool call."""
        text = '{"name": "bash"}'
        calls, cleaned = extract_tool_calls_from_text(text)
        assert calls == []

    def test_returns_toolcall_instances(self):
        text = '{"name": "file_read", "arguments": {"path": "x.py"}}'
        calls, _ = extract_tool_calls_from_text(text)
        assert all(isinstance(c, ToolCall) for c in calls)

    def test_generated_id_is_unique(self):
        text = (
            '{"name": "file_read", "arguments": {"path": "a.py"}}\n'
            '{"name": "file_read", "arguments": {"path": "b.py"}}'
        )
        calls, _ = extract_tool_calls_from_text(text)
        if len(calls) >= 2:
            assert calls[0].id != calls[1].id

    def test_multiline_tool_call(self):
        """Tool call JSON spread across multiple lines."""
        text = (
            "Tool Calls: [\n"
            "  {\n"
            '    "id": "call_1",\n'
            '    "type": "function",\n'
            '    "function": {\n'
            '      "name": "file_write",\n'
            '      "arguments": {\n'
            '        "path": "out.py",\n'
            '        "content": "print(1)"\n'
            "      }\n"
            "    }\n"
            "  }\n"
            "]"
        )
        calls, _ = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "file_write"
        assert calls[0].arguments == {"path": "out.py", "content": "print(1)"}

    def test_content_with_brackets(self):
        """Tool call whose arguments contain [] should still parse correctly.

        This is the root cause of code-gen-003 and long-horizon-001 failures:
        content like ``self.order = []`` causes a naive regex ``\\[.*?\\]``
        to stop at the first ``]`` inside the JSON string, truncating the match.
        """
        text = (
            'Tool Calls: [ { "id": "call_lru", "type": "function", '
            '"function": { "name": "file_write", "arguments": { '
            '"path": "src/solution.py", '
            '"content": "class LRU:\\n    def __init__(self):\\n'
            '        self.order = []\\n        self.cache = {}\\n"'
            " } } } ]"
        )
        calls, cleaned = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "file_write"
        assert calls[0].arguments["path"] == "src/solution.py"
        assert "self.order = []" in calls[0].arguments["content"]
        assert "Tool Calls:" not in cleaned

    def test_content_with_nested_braces_and_brackets(self):
        """Content containing both {} and [] should still parse."""
        text = (
            'Tool Calls: [ { "id": "call_1", "type": "function", '
            '"function": { "name": "file_write", "arguments": { '
            '"path": "test.py", '
            '"content": "data = {\\"a\\": [1, 2, 3]}\\nresult = data[\\"a\\"]"'
            " } } } ]"
        )
        calls, _ = extract_tool_calls_from_text(text)
        assert len(calls) == 1
        assert calls[0].name == "file_write"

    def test_multiple_tool_calls_with_brackets_in_content(self):
        """Multiple tool calls where content has brackets."""
        text = (
            "Tool Calls: [ "
            '{ "id": "c1", "type": "function", '
            '"function": { "name": "file_write", "arguments": { '
            '"path": "a.py", "content": "x = [1, 2]\\n" } } }, '
            '{ "id": "c2", "type": "function", '
            '"function": { "name": "bash", "arguments": { '
            '"command": "pytest" } } } ]'
        )
        calls, _ = extract_tool_calls_from_text(text)
        assert len(calls) == 2
        assert calls[0].name == "file_write"
        assert calls[1].name == "bash"
