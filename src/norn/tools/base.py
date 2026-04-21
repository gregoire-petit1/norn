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
