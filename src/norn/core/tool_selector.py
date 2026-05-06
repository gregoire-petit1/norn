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
            recent_context: Optional text from recent assistant messages.

        Returns:
            List of selected tool names (preserves order from available_tools).
        """
        if not self.enabled:
            return available_tools

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

        # Preserve original ordering from available_tools
        return [name for name in available_tools if name in selected]
