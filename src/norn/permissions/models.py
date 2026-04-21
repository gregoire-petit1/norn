"""Permission system models for Norn."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class PermissionDecisionReason(StrEnum):
    """Canonical taxonomy for the ``reason`` field on :class:`PermissionDecision`.

    Emitted in ``permission.decision`` structured events. Consumers
    (dashboards, log filters) should match against these values.

    Granularity rationale: the ``mode`` and ``risk_level`` fields in the
    structured event already carry the contextual nuance, so the ``reason``
    field describes the *decision class* only. In particular, both
    auto/interactive auto-approve paths collapse to ``auto_approved`` —
    the (mode, risk, reason) triple is what downstream filters should key on.
    """

    YOLO = "yolo"
    AUTO_APPROVED = "auto_approved"
    USER_APPROVED = "user_approved"
    USER_DENIED = "user_denied"
    DESTRUCTIVE_DENIED = "destructive_denied"
    PROMPT_REQUIRED_NO_HANDLER = "prompt_required_no_handler"


class PermissionRequest(BaseModel):
    """Request to check permission for a tool invocation."""

    tool_name: str
    risk_level: str  # "low", "medium", "high"
    arguments: dict[str, Any] = {}


class PermissionDecision(BaseModel):
    """Result of a permission check."""

    approved: bool
    reason: str | None = None
    escalated_risk: str | None = None  # If risk was escalated from base level
