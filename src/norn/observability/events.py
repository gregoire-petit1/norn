"""Event name constants for structured logging."""

from __future__ import annotations

from enum import StrEnum


class EventName(StrEnum):
    """Canonical event names emitted by Norn's observability layer.

    Treat values as a stable contract: downstream log consumers
    (jq queries, dashboards, metrics) depend on exact strings.
    """

    AGENT_RUN = "agent.run"
    LLM_COMPLETE = "llm.complete"
    TOOL_CALL = "tool.call"
    PERMISSION_DECISION = "permission.decision"
    ROUTING_DECISION = "routing.decision"
    FALLBACK = "fallback"
