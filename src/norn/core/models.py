"""Core message and response models for the Norn agent."""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, computed_field


class Role(StrEnum):
    """Message role in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


class TokenUsage(BaseModel):
    """Token usage statistics."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    # Prompt-cache metrics (Anthropic-style; 0 when the provider reports none).
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0


class Message(BaseModel):
    """A single message in the conversation history."""

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class LLMResponse(BaseModel):
    """Response from an LLM completion call."""

    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None
    latency_ms: int | None = None
    model: str | None = None

    @computed_field
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class StreamChunk(BaseModel):
    """A chunk from a streaming LLM response."""

    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    done: bool = False
    usage: TokenUsage | None = None


class EventType(StrEnum):
    """Types of events yielded by the streaming agent loop."""

    TEXT_DELTA = "text_delta"
    TOOL_START = "tool_start"
    TOOL_END = "tool_end"
    DONE = "done"


class AgentEvent(BaseModel):
    """Event yielded by AgentLoop.run_stream()."""

    type: EventType
    content: str | None = None
    tool_name: str | None = None
    tool_args: str | None = None
    tool_result: str | None = None
    duration_ms: int | None = None
    success: bool | None = None
    usage: TokenUsage | None = None
    latency_ms: int | None = None
    model: str | None = None
