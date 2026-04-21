"""Core agent loop for Norn."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from norn.core.models import LLMResponse, Message, Role, ToolCall
from norn.observability import EventName, get_logger
from norn.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider
    from norn.memory.session_logger import SessionLogger
    from norn.memory.store import MemoryStore
    from norn.permissions.checker import PermissionChecker
    from norn.tools.registry import ToolRegistry


_log = get_logger(__name__)


class AgentLoop:
    """The main agent loop: message -> LLM -> tool calls -> repeat."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str = "You are Norn, a helpful coding agent.",
        cwd: str = ".",
        permission_checker: PermissionChecker | None = None,
        memory_store: MemoryStore | None = None,
        session_logger: SessionLogger | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cwd=cwd)
        self.history: list[Message] = []
        self.permission_checker = permission_checker
        self.memory_store = memory_store
        self.session_logger = session_logger
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
        return prompt

    async def run(self, user_input: str) -> LLMResponse:
        """Run one turn of the agent loop."""
        start = time.monotonic()
        _log.info(EventName.AGENT_RUN, phase="start")
        success = False
        error: str | None = None
        try:
            result = await self._run_impl(user_input)
            success = True
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            _log.info(
                EventName.AGENT_RUN,
                phase="end",
                duration_ms=int((time.monotonic() - start) * 1000),
                success=success,
                error=error,
            )

    async def _run_impl(self, user_input: str) -> LLMResponse:
        self.user_message_count += 1
        self.history.append(Message(role=Role.USER, content=user_input))

        messages = [
            Message(role=Role.SYSTEM, content=self._build_system_prompt()),
            *self.history,
        ]

        for _round in range(self.MAX_TOOL_ROUNDS):
            response = await self.llm.complete(
                messages=messages,
                tools=self.registry.get_schemas() or None,
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
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output or result.error or "",
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)
                self.history.append(tool_msg)

        # Safety: max rounds reached
        final = LLMResponse(content="[Max tool rounds reached]")
        self.history.append(Message(role=Role.ASSISTANT, content=final.content))
        return final

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call, with optional permission check."""
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {call.name}")

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
                return ToolResult(error=f"Permission denied: {reason}")

        try:
            input_obj = tool.input_model(**call.arguments)
            return await tool.execute(input_obj, self.ctx)
        except Exception as e:
            return ToolResult(error=f"Tool execution error: {e}")
