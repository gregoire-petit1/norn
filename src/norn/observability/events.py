"""Event name constants and schema contract for structured logging.

Error schema
------------
Any event carrying an exception MUST expose the two following fields
(and ONLY these two — the conflated ``error`` field is intentionally
dropped):

- ``error_type``: ``str`` — exception class name, e.g. ``"RuntimeError"``.
- ``error_message``: ``str`` — ``str(exc)``, the human-readable message.

Rationale: separating class from message lets downstream consumers group
by type (metrics, alerting) while retaining the message for debugging.
A single ``error`` field collapsing both prevents structured filtering.

All lifecycle events (AGENT_RUN end, LLM_COMPLETE) additionally carry:

- ``success``: ``bool`` — ``True`` on happy path, ``False`` on exception.
- ``duration_ms``/``latency_ms``: ``int`` — elapsed wall time.

On success, ``error_type``/``error_message`` are absent (not null).
"""

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
