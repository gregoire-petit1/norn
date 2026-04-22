"""LLM provider abstraction for Norn."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Awaitable, Callable

import logging as _stdlib_logging

import litellm

from norn.core.models import (
    LLMResponse,
    Message,
    StreamChunk,
    TokenUsage,
    ToolCall,
)
from norn.observability import EventName, get_logger, measure_and_log

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


# --------------------------------------------------------------------------- #
# Prompt caching (Phase 9 v2 — Workstream G)
# --------------------------------------------------------------------------- #

_CACHE_ELIGIBLE_PREFIXES = (
    "anthropic/",
    "openrouter/anthropic/",
    "openai/gpt-4o",
    "openai/gpt-4.1",
)


def _supports_prompt_cache(model: str) -> bool:
    """Return True if ``model`` is known to honour ``cache_control`` markers.

    Conservative gate: better to skip caching on a supporting model than to
    send unrecognised fields to a fragile provider. Anthropic uses explicit
    ``cache_control: {"type": "ephemeral"}`` markers; OpenAI's automatic
    caching ignores them harmlessly, so they're safe to leave on for the
    listed OpenAI families.
    """
    return model.startswith(_CACHE_ELIGIBLE_PREFIXES)


def _apply_cache_markers(
    messages: list[dict],
    tools: list[dict] | None,
    enabled: bool,
) -> tuple[list[dict], list[dict] | None]:
    """Tag system prompt + last tool schema with ``cache_control: ephemeral``.

    No-op when ``enabled`` is False. Caller must also gate on
    :func:`_supports_prompt_cache` to avoid sending markers to providers
    that don't understand them.

    Returns a (messages, tools) tuple with the marked copies; the originals
    are not mutated.
    """
    if not enabled:
        return messages, tools

    new_messages = list(messages)
    if new_messages and new_messages[0].get("role") == "system":
        sys_msg = dict(new_messages[0])
        content = sys_msg.get("content", "")
        if isinstance(content, str):
            sys_msg["content"] = [
                {
                    "type": "text",
                    "text": content,
                    "cache_control": {"type": "ephemeral"},
                }
            ]
        new_messages[0] = sys_msg

    new_tools = tools
    if tools:
        new_tools = list(tools)
        new_tools[-1] = {
            **new_tools[-1],
            "cache_control": {"type": "ephemeral"},
        }

    return new_messages, new_tools


class LiteLLMProvider:
    """LLM provider using litellm for universal model support."""

    def __init__(
        self,
        model: str,
        api_base: str | None = None,
        *,
        completion_fn: Callable[..., Awaitable[Any]] | None = None,
        prompt_cache: bool = True,
    ) -> None:
        self.model = model
        self.api_base = api_base
        # When no override is given, resolve ``litellm.acompletion`` lazily at
        # call time. This preserves backward compatibility with tests that
        # ``patch("litellm.acompletion", ...)`` after the provider is built.
        # Tests wanting per-instance isolation (e.g. concurrent tasks) should
        # pass ``completion_fn=`` explicitly.
        self._completion_fn = completion_fn
        # Gate at construction: drop the flag immediately if the model is
        # not on the allowlist, so the hot path stays a single bool check.
        self._prompt_cache = prompt_cache and _supports_prompt_cache(model)
        # Suppress litellm logging noise
        litellm.suppress_debug_info = True
        _stdlib_logging.getLogger("LiteLLM").setLevel(_stdlib_logging.WARNING)

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        # Guard: empty messages is a programmer error and the underlying
        # provider behaviour ranges from "4xx" to "silent empty completion"
        # to "hang". Surface it deterministically before any network call.
        if not messages:
            raise ValueError("messages cannot be empty")

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

        # Apply prompt-cache markers if the model is eligible. Operates on
        # the dict copies in ``kwargs`` so the original Message objects stay
        # untouched and observable.
        if self._prompt_cache:
            marked_messages, marked_tools = _apply_cache_markers(
                kwargs["messages"],
                kwargs.get("tools"),
                enabled=True,
            )
            kwargs["messages"] = marked_messages
            if marked_tools is not None:
                kwargs["tools"] = marked_tools

        async with measure_and_log(
            _log,
            EventName.LLM_COMPLETE,
            duration_field="latency_ms",
            provider=_provider_from_model(self.model),
            model=self.model,
        ) as event:
            completion = self._completion_fn or litellm.acompletion
            _start = __import__("time").monotonic()
            response = await completion(**kwargs)

            # Guard against malformed / minimal responses:
            # - ``response.choices`` may be empty (provider refusal, mocks).
            # - ``response.usage`` may be ``None`` (some Ollama configs).
            choice = response.choices[0] if response.choices else None
            usage = getattr(response, "usage", None)

            if usage is not None:
                token_usage = TokenUsage(
                    prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                    completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
                    total_tokens=getattr(usage, "total_tokens", 0) or 0,
                )
            else:
                token_usage = TokenUsage(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )

            event["prompt_tokens"] = token_usage.prompt_tokens
            event["completion_tokens"] = token_usage.completion_tokens
            event["total_tokens"] = token_usage.total_tokens
            event["cache_read_tokens"] = (
                (getattr(usage, "cache_read_input_tokens", 0) or 0) if usage is not None else 0
            )
            event["cache_creation_tokens"] = (
                (getattr(usage, "cache_creation_input_tokens", 0) or 0) if usage is not None else 0
            )
            event["finish_reason"] = (
                getattr(choice, "finish_reason", None) if choice is not None else None
            )

            content = choice.message.content if choice is not None else None
            tool_calls = _parse_tool_calls(choice.message.tool_calls) if choice is not None else []
            _elapsed_ms = int((__import__("time").monotonic() - _start) * 1000)
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                usage=token_usage,
                latency_ms=_elapsed_ms,
                model=self.model,
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
