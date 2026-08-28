"""Tool base types and protocols for Norn."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, computed_field


class RiskLevel(StrEnum):
    """Risk classification for tool actions."""

    LOW = "low"  # Read-only operations
    MEDIUM = "medium"  # Writes with potential undo
    HIGH = "high"  # Destructive or irreversible


class ToolErrorType(StrEnum):
    """Semantic error categories for ``ToolResult.error_type``.

    Used by tool implementations to tag failures with stable, queryable
    categories. Mirrors :class:`norn.permissions.models.PermissionDecisionReason`
    introduced in Phase 9 v1.

    Values are surfaced verbatim in JSONL ``tool.call`` events; downstream
    dashboards and log filters key on them, so renaming a member is a
    breaking change.
    """

    FILE_NOT_FOUND = "FileNotFound"
    PERMISSION_DENIED = "PermissionDenied"
    INVALID_ARGUMENT = "InvalidArgument"
    TIMEOUT = "Timeout"
    NETWORK_ERROR = "NetworkError"
    HTTP_ERROR = "HttpError"
    PARSE_ERROR = "ParseError"
    NOT_SUPPORTED = "NotSupported"
    EXECUTION_ERROR = "ExecutionError"
    RESOURCE_EXHAUSTED = "ResourceExhausted"
    # SOTA v2 (workstream C): confinement requested but no sandbox mechanism
    # is available — execution refused (fail-closed), never run unconfined.
    SANDBOX_UNAVAILABLE = "SandboxUnavailable"


class ToolContext(BaseModel):
    """Context passed to tool execution."""

    cwd: str = "."


class ToolResult(BaseModel):
    """Result of a tool execution."""

    output: str | None = None
    error: str | None = None
    error_type: str | None = None  # categorises the failure for observability

    @computed_field
    @property
    def is_error(self) -> bool:
        return self.error is not None


@runtime_checkable
class Tool(Protocol):
    """Protocol that all tools must implement."""

    name: str
    description: str
    risk_level: RiskLevel
    input_model: type[BaseModel]

    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult: ...
