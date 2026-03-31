"""Shared utilities for Norn."""

from __future__ import annotations

import re


def extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code blocks.

    Handles patterns like:
    - Raw JSON
    - ```json\n{...}\n```
    - ```\n{...}\n```
    - Text before/after code blocks
    """
    # Try to find a JSON code block first
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()
