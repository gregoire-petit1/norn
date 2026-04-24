"""User-friendly error formatting for CLI."""

from __future__ import annotations

import re

_RATE_LIMIT_PATTERNS = (
    re.compile(r"429", re.IGNORECASE),
    re.compile(r"rate.?limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
)

_CONNECTION_PATTERNS = (
    re.compile(r"connection.?(refused|error|reset)", re.IGNORECASE),
    re.compile(r"unreachable", re.IGNORECASE),
)


def format_llm_error(exc: Exception) -> str:
    """Format an LLM error into a user-friendly message."""
    msg = str(exc)

    for pattern in _RATE_LIMIT_PATTERNS:
        if pattern.search(msg):
            return "Rate limit reached. Wait a few minutes or use /model to switch provider."

    for pattern in _CONNECTION_PATTERNS:
        if pattern.search(msg):
            return "Cannot connect to the LLM provider. Check that it's running."

    if "timeout" in msg.lower():
        return "Request timed out. The model may be overloaded."

    # Fallback: clean up the raw error
    return f"LLM error: {msg}"
