"""Core agent loop for Norn."""

from __future__ import annotations

import contextlib
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Callable

from norn.core.models import (
    AgentEvent,
    EventType,
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    ToolCall,
)
from norn.core.prompts import AGENT_SYSTEM_PROMPT
from norn.core.tool_call_extractor import extract_tool_calls_from_text
from norn.core.truncation import truncate_tool_output
from norn.observability import EventName, get_logger, measure_and_log
from norn.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider
    from norn.memory.session_logger import SessionLogger
    from norn.memory.store import MemoryStore
    from norn.permissions.checker import PermissionChecker
    from norn.tools.registry import ToolRegistry

# Callback type: (tool_name, args_summary, duration_ms, success) -> None
ToolProgressCallback = Callable[[str, str, int, bool], None]


_log = get_logger(__name__)


class AgentLoop:
    """The main agent loop: message -> LLM -> tool calls -> repeat."""

    DEFAULT_MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        cwd: str = ".",
        permission_checker: PermissionChecker | None = None,
        memory_store: MemoryStore | None = None,
        session_logger: SessionLogger | None = None,
        on_tool_progress: ToolProgressCallback | None = None,
        max_tool_rounds: int | None = None,
        minify_tool_schemas: bool = True,
        max_tool_result_chars: int = 8000,
        env_bootstrap: bool = True,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cwd=cwd)
        self.history: list[Message] = []
        self.permission_checker = permission_checker
        self.memory_store = memory_store
        self.session_logger = session_logger
        self._on_tool_progress = on_tool_progress
        self._max_tool_rounds = max_tool_rounds or self.DEFAULT_MAX_TOOL_ROUNDS
        self._minify_tool_schemas = minify_tool_schemas
        self._max_tool_result_chars = max_tool_result_chars
        # Environment bootstrap
        self._env_snapshot: str | None = None
        if env_bootstrap:
            from norn.core.env_bootstrap import scan_environment

            snapshot = scan_environment(cwd)
            self._env_snapshot = snapshot.render()
        # Session stats
        self.user_message_count = 0
        self.tool_call_count = 0

    def _build_system_prompt(self) -> str:
        """Build system prompt with optional memory injection."""
        prompt = self.system_prompt
        if self.memory_store is not None:
            memory_content = self.memory_store.read_memory()
            if memory_content.strip():
                prompt += "\n\n## Persistent Memory\n\n" + memory_content
        if self._env_snapshot:
            prompt += "\n\n" + self._env_snapshot
        return prompt

    async def run(self, user_input: str) -> LLMResponse:
        """Run one turn of the agent loop."""
        # The ``agent.run`` start marker is additive; a logging failure here
        # must not prevent the turn from executing.
        with contextlib.suppress(Exception):
            _log.info(EventName.AGENT_RUN, phase="start")
        async with measure_and_log(_log, EventName.AGENT_RUN, phase="end"):
            return await self._run_impl(user_input)

    async def _run_impl(self, user_input: str) -> LLMResponse:
        self.user_message_count += 1
        self.history.append(Message(role=Role.USER, content=user_input))

        messages = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            *self.history,
        ]

        for _round in range(self._max_tool_rounds):
            response = await self.llm.complete(
                messages=messages,
                tools=self.registry.get_schemas(minify=self._minify_tool_schemas) or None,
            )

            if not response.has_tool_calls:
                # Fallback: try to extract tool calls from text output
                if response.content:
                    extracted, cleaned = extract_tool_calls_from_text(response.content)
                    if extracted:
                        with contextlib.suppress(Exception):
                            _log.info(
                                "tool_call_extraction",
                                count=len(extracted),
                                tool_names=[c.name for c in extracted],
                            )
                        response = LLMResponse(
                            content=cleaned or None,
                            tool_calls=extracted,
                            usage=response.usage,
                            latency_ms=response.latency_ms,
                            model=response.model,
                        )

            if not response.has_tool_calls:
                self.history.append(Message(role=Role.ASSISTANT, content=response.content))
                return response

            # Process tool calls
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)
            self.history.append(assistant_msg)

            for call in response.tool_calls:
                self.tool_call_count += 1
                result = await self._execute_tool(call)
                raw_content = result.output or result.error or ""
                content = truncate_tool_output(raw_content, self._max_tool_result_chars)
                tool_msg = Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)
                self.history.append(tool_msg)

        # Safety: max rounds reached
        final = LLMResponse(content="[Max tool rounds reached]")
        self.history.append(Message(role=Role.ASSISTANT, content=final.content))
        return final

    async def run_stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run one turn of the agent loop, yielding events for real-time rendering.

        Mirrors ``_run_impl`` logic but yields ``AgentEvent`` objects
        progressively so the CLI can render text as it arrives and show
        tool progress in real time.
        """
        self.user_message_count += 1
        self.history.append(Message(role=Role.USER, content=user_input))

        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            *self.history,
        ]

        for _round in range(self._max_tool_rounds):
            # --- Stream from LLM ---
            accumulated_content = ""
            accumulated_tool_calls: list[ToolCall] = []
            final_usage = None

            async for chunk in self.llm.stream(
                messages=messages,
                tools=self.registry.get_schemas(minify=self._minify_tool_schemas) or None,
            ):
                if chunk.content:
                    accumulated_content += chunk.content
                    yield AgentEvent(type=EventType.TEXT_DELTA, content=chunk.content)

                if chunk.done:
                    if chunk.tool_calls:
                        accumulated_tool_calls = chunk.tool_calls
                    if chunk.usage:
                        final_usage = chunk.usage

            # --- Fallback: extract tool calls from text output ---
            tool_calls = accumulated_tool_calls
            if not tool_calls and accumulated_content:
                extracted, cleaned = extract_tool_calls_from_text(accumulated_content)
                if extracted:
                    with contextlib.suppress(Exception):
                        _log.info(
                            "tool_call_extraction",
                            count=len(extracted),
                            tool_names=[c.name for c in extracted],
                        )
                    tool_calls = extracted
                    accumulated_content = cleaned or ""

            # --- No tool calls → turn is done ---
            if not tool_calls:
                self.history.append(
                    Message(role=Role.ASSISTANT, content=accumulated_content or None)
                )
                yield AgentEvent(type=EventType.DONE, usage=final_usage)
                return

            # --- Process tool calls ---
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=accumulated_content or None,
                tool_calls=tool_calls,
            )
            messages.append(assistant_msg)
            self.history.append(assistant_msg)

            for call in tool_calls:
                self.tool_call_count += 1
                yield AgentEvent(
                    type=EventType.TOOL_START,
                    tool_name=call.name,
                    tool_args=self._summarize_tool_args(call),
                )

                start = time.monotonic()
                result = await self._execute_tool(call)
                duration_ms = int((time.monotonic() - start) * 1000)

                raw_content = result.output or result.error or ""
                content = truncate_tool_output(raw_content, self._max_tool_result_chars)

                tool_msg = Message(
                    role=Role.TOOL,
                    content=content,
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)
                self.history.append(tool_msg)

                yield AgentEvent(
                    type=EventType.TOOL_END,
                    tool_name=call.name,
                    tool_args=self._summarize_tool_args(call),
                    tool_result=content,
                    duration_ms=duration_ms,
                    success=result.error is None,
                )

        # Safety: max tool rounds reached
        self.history.append(Message(role=Role.ASSISTANT, content="[Max tool rounds reached]"))
        yield AgentEvent(type=EventType.DONE)

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call, with optional permission check.

        Emits a ``tool.call`` structured event on every invocation (including
        unknown-tool and permission-denied paths) with ``tool_name``,
        ``duration_ms``, ``success``, and — on failure — the harmonised
        ``error_type`` / ``error_message`` pair.

        Direct emission is used rather than :func:`measure_and_log`: tool
        failures are never raised here (they are wrapped into
        ``ToolResult(error=...)``), so the helper's non-raise branch would
        always report ``success=True`` and the caller cannot override it.
        The :func:`contextlib.suppress` wrapper preserves fail-open
        semantics.
        """
        start = time.monotonic()
        result = await self._execute_tool_inner(call)
        duration_ms = int((time.monotonic() - start) * 1000)
        payload: dict[str, object] = {
            "tool_name": call.name,
            "duration_ms": duration_ms,
            "success": result.error is None,
        }
        if result.error is not None:
            payload["error_type"] = result.error_type or "ToolExecutionError"
            payload["error_message"] = result.error
        with contextlib.suppress(Exception):
            if result.error is None:
                _log.info(EventName.TOOL_CALL, **payload)
            else:
                _log.error(EventName.TOOL_CALL, **payload)
        # Notify CLI progress callback
        if self._on_tool_progress is not None:
            with contextlib.suppress(Exception):
                args_summary = self._summarize_tool_args(call)
                self._on_tool_progress(call.name, args_summary, duration_ms, result.error is None)
        return result

    @staticmethod
    def _summarize_tool_args(call: ToolCall) -> str:
        """Produce a short summary of tool arguments for display."""
        args = call.arguments
        # Common patterns: bash has "command", file tools have "path"
        if "command" in args:
            cmd = str(args["command"])
            return cmd[:60] + ("..." if len(cmd) > 60 else "")
        if "path" in args:
            return str(args["path"])
        if "file_path" in args:
            return str(args["file_path"])
        if "pattern" in args:
            return str(args["pattern"])
        # Fallback: first string arg value
        for v in args.values():
            if isinstance(v, str):
                return v[:50] + ("..." if len(str(v)) > 50 else "")
        return ""

    async def _execute_tool_inner(self, call: ToolCall) -> ToolResult:
        """Core tool resolution + invocation. Never raises; returns ToolResult."""
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {call.name}", error_type="UnknownTool")

        # Permission check
        if self.permission_checker is not None:
            from norn.permissions.models import PermissionRequest

            request = PermissionRequest(
                tool_name=call.name,
                risk_level=tool.risk_level.value,
                arguments=call.arguments,
            )
            decision = await self.permission_checker.check(request)
            if not decision.approved:
                reason = decision.reason or "Permission denied"
                return ToolResult(
                    error=f"Permission denied: {reason}", error_type="PermissionDenied"
                )

        try:
            input_obj = tool.input_model(**call.arguments)
            return await tool.execute(input_obj, self.ctx)
        except Exception as e:
            return ToolResult(error=f"Tool execution error: {e}", error_type=type(e).__name__)
