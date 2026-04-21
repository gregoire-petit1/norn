"""LLM provider abstraction for Norn."""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

import litellm

from norn.core.models import (
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from norn.observability import EventName, get_logger

_log = get_logger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM backends must implement."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]: ...


def build_tool_schemas(tools: list[dict]) -> list[dict]:
    """Convert tool definitions to OpenAI-compatible function schemas."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert Message objects to dicts for litellm."""
    result = []
    for msg in messages:
        d: dict = {"role": msg.role.value, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        result.append(d)
    return result


def _parse_tool_calls(raw_tool_calls: list | None) -> list[ToolCall]:
    """Parse tool calls from litellm response."""
    if not raw_tool_calls:
        return []
    calls = []
    for tc in raw_tool_calls:
        arguments = tc.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=arguments))
    return calls


def _provider_from_model(model: str) -> str:
    """Extract provider name from a litellm-style model id.

    Litellm uses a ``<provider>/<model>`` convention (e.g. ``ollama/llama3``,
    ``openrouter/x``). Unprefixed names default to ``openai``.
    """
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


class LiteLLMProvider:
    """LLM provider using litellm for universal model support."""

    def __init__(self, model: str, api_base: str | None = None):
        self.model = model
        self.api_base = api_base
        # Suppress litellm logging noise
        litellm.suppress_debug_info = True

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        tool_schemas = build_tool_schemas(tools or [])
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        start = time.monotonic()
        try:
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:
            _log.error(
                EventName.LLM_COMPLETE,
                provider=_provider_from_model(self.model),
                model=self.model,
                latency_ms=int((time.monotonic() - start) * 1000),
                error=str(exc),
                error_type=type(exc).__name__,
            )
            raise

        latency_ms = int((time.monotonic() - start) * 1000)
        choice = response.choices[0]
        usage = getattr(response, "usage", None)

        _log.info(
            EventName.LLM_COMPLETE,
            provider=_provider_from_model(self.model),
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            latency_ms=latency_ms,
            finish_reason=getattr(choice, "finish_reason", None),
        )

        return LLMResponse(
            content=choice.message.content,
            tool_calls=_parse_tool_calls(choice.message.tool_calls),
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict = {
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        tool_schemas = build_tool_schemas(tools or [])
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta
            yield StreamChunk(
                content=delta.content if hasattr(delta, "content") else None,
                done=chunk.choices[0].finish_reason is not None,
            )
