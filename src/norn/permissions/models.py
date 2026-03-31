"""Permission system models for Norn."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


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
