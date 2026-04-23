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

# Matches "Tool Calls: [...]" with the JSON array (possibly multiline).
_TOOL_CALLS_BLOCK_RE = re.compile(
    r"Tool\s+Calls:\s*(\[.*?\])",
    re.DOTALL,
)

# Matches a standalone JSON object on its own line(s) that looks like a tool call.
# Must have "name" and "arguments" keys.
_FLAT_TOOL_CALL_RE = re.compile(
    r'(?<![`])(\{[^{}]*"name"\s*:\s*"[^"]+?"[^{}]*"arguments"\s*:\s*\{[^}]*\}[^{}]*\})',
    re.DOTALL,
)


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
    """Try to extract OpenAI-style tool calls from text."""
    match = _TOOL_CALLS_BLOCK_RE.search(stripped)
    if not match:
        return [], original

    json_str = match.group(1)
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

    # Remove the "Tool Calls: [...]" block from original text
    cleaned = _TOOL_CALLS_BLOCK_RE.sub("", original).strip()
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
