"""Fallback parser for tool calls emitted as text by the LLM.

Some models (e.g. qwen3-coder) sometimes emit tool calls as inline JSON text
instead of using the function calling API.  This module extracts those calls
so the agent loop can execute them.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from norn.core.models import ToolCall

# Matches the "Tool Calls:" prefix to locate the start of the JSON array.
_TOOL_CALLS_PREFIX_RE = re.compile(r"Tool\s+Calls:\s*")

# Matches a standalone JSON object on its own line(s) that looks like a tool call.
# Must have "name" and "arguments" keys.
_FLAT_TOOL_CALL_RE = re.compile(
    r'(?<![`])(\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\})',
    re.DOTALL,
)


def _find_balanced_bracket(text: str, start: int) -> int | None:
    """Find the closing ``]`` that balances the opening ``[`` at *start*.

    Handles brackets inside JSON strings (skipping ``\\"`` escapes).
    Returns the index of the matching ``]`` or ``None`` if not found.
    """
    if start >= len(text) or text[start] != "[":
        return None

    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_string = False
        else:
            if ch == '"':
                in_string = True
            elif ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return i
        i += 1
    return None


def extract_tool_calls_from_text(text: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls from text, return (calls, cleaned_text).

    Tries two patterns:
      1. OpenAI-style: ``Tool Calls: [ { "id": ..., "function": { ... } } ]``
      2. Flat JSON:    ``{"name": "bash", "arguments": {...}}``

    Returns an empty list if no valid tool calls are found.
    Never raises on malformed input.
    """
    if not text:
        return [], ""

    # Skip extraction from markdown code blocks
    stripped = _strip_code_blocks(text)

    # Try OpenAI-style first (higher confidence)
    calls, cleaned = _try_openai_pattern(text, stripped)
    if calls:
        return calls, cleaned

    # Try flat JSON pattern
    calls, cleaned = _try_flat_pattern(text, stripped)
    if calls:
        return calls, cleaned

    return [], text


def _strip_code_blocks(text: str) -> str:
    """Remove markdown fenced code blocks from text for pattern matching."""
    return re.sub(r"```[\s\S]*?```", "", text)


def _try_openai_pattern(original: str, stripped: str) -> tuple[list[ToolCall], str]:
    """Try to extract OpenAI-style tool calls from text.

    Uses bracket-matching instead of regex to handle JSON strings that
    contain ``[`` and ``]`` characters (e.g. ``self.order = []``).
    """
    prefix_match = _TOOL_CALLS_PREFIX_RE.search(stripped)
    if not prefix_match:
        return [], original

    # Find the start of the JSON array
    array_start = prefix_match.end()
    # Skip whitespace to find the opening bracket
    while array_start < len(stripped) and stripped[array_start] in " \t\n\r":
        array_start += 1
    if array_start >= len(stripped) or stripped[array_start] != "[":
        return [], original

    # Use bracket-matching to find the end
    array_end = _find_balanced_bracket(stripped, array_start)
    if array_end is None:
        return [], original

    json_str = stripped[array_start : array_end + 1]
    try:
        items = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return [], original

    if not isinstance(items, list):
        return [], original

    calls = []
    for item in items:
        call = _parse_openai_item(item)
        if call is not None:
            calls.append(call)

    if not calls:
        return [], original

    # Remove the "Tool Calls: [...]" block from original text.
    # Find the same prefix in the original (not stripped) text.
    orig_prefix = _TOOL_CALLS_PREFIX_RE.search(original)
    if orig_prefix:
        arr_start_orig = orig_prefix.end()
        while arr_start_orig < len(original) and original[arr_start_orig] in " \t\n\r":
            arr_start_orig += 1
        arr_end_orig = _find_balanced_bracket(original, arr_start_orig)
        if arr_end_orig is not None:
            cleaned = (original[: orig_prefix.start()] + original[arr_end_orig + 1 :]).strip()
            return calls, cleaned

    # Fallback: remove via prefix match on original
    cleaned = _TOOL_CALLS_PREFIX_RE.sub("", original).strip()
    return calls, cleaned


def _parse_openai_item(item: dict[str, Any]) -> ToolCall | None:
    """Parse a single OpenAI-style tool call dict."""
    if not isinstance(item, dict):
        return None

    func = item.get("function")
    if not isinstance(func, dict):
        return None

    name = func.get("name")
    arguments = func.get("arguments")
    if not name or not isinstance(arguments, dict):
        return None

    call_id = item.get("id") or f"extracted_{uuid.uuid4().hex[:12]}"
    return ToolCall(id=call_id, name=name, arguments=arguments)


def _try_flat_pattern(original: str, stripped: str) -> tuple[list[ToolCall], str]:
    """Try to extract flat JSON tool calls from text."""
    matches = _FLAT_TOOL_CALL_RE.findall(stripped)
    if not matches:
        return [], original

    calls: list[ToolCall] = []
    for json_str in matches:
        call = _parse_flat_item(json_str)
        if call is not None:
            calls.append(call)

    if not calls:
        return [], original

    # Remove matched JSON blocks from original text
    cleaned = original
    for json_str in matches:
        cleaned = cleaned.replace(json_str, "")
    cleaned = cleaned.strip()

    return calls, cleaned


def _parse_flat_item(json_str: str) -> ToolCall | None:
    """Parse a flat JSON tool call string."""
    try:
        data = json.loads(json_str)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(data, dict):
        return None

    name = data.get("name")
    arguments = data.get("arguments")

    if not name or not isinstance(name, str):
        return None
    if not isinstance(arguments, dict):
        return None

    call_id = data.get("id") or f"extracted_{uuid.uuid4().hex[:12]}"
    return ToolCall(id=call_id, name=name, arguments=arguments)
