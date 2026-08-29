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
from norn.core.thread_ledger import ThreadInvariantError, ThreadLedger
from norn.core.tool_call_extractor import extract_tool_calls_from_text
from norn.core.turn_budget import TurnBudgetTracker
from norn.observability import EventName, get_logger, measure_and_log
from norn.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from norn.core.context import ContextManager
    from norn.core.llm import LLMProvider
    from norn.core.task_notes import TaskNotes
    from norn.core.tool_selector import ToolSelector
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
        max_turn_output_chars: int = 30000,
        env_bootstrap: bool = True,
        repo_map: bool = True,
        repo_map_max_chars: int = 2000,
        repo_map_languages: list[str] | None = None,
        repo_map_exclude: list[str] | None = None,
        context_manager: ContextManager | None = None,
        tool_selector: ToolSelector | None = None,
        stable_prompt: bool = False,
        thread_invariants: bool = True,
        task_notes: TaskNotes | None = None,
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
        self._turn_budget = TurnBudgetTracker(max_chars_per_turn=max_turn_output_chars)
        self._context_manager = context_manager
        self._tool_selector = tool_selector
        if tool_selector is not None:
            # Dynamic tool selection varies the `tools` array per turn; on
            # providers where tools precede the system prompt in the cache
            # prefix (Anthropic), this defeats prompt caching entirely.
            with contextlib.suppress(Exception):
                _log.warning("dynamic_tools_breaks_prompt_cache")
        # Environment bootstrap
        self._env_snapshot: str | None = None
        if env_bootstrap:
            from norn.core.env_bootstrap import scan_environment

            snapshot = scan_environment(cwd)
            self._env_snapshot = snapshot.render()
        # Repo map (W3.1)
        self._repo_map: str | None = None
        if repo_map:
            from norn.core.repo_map import RepoMap

            rm = RepoMap(
                max_chars=repo_map_max_chars,
                languages=repo_map_languages or ["python", "typescript"],
                exclude_patterns=repo_map_exclude or [],
            )
            self._repo_map = rm.scan(cwd)
        # Session stats
        self.user_message_count = 0
        self.tool_call_count = 0
        # W1.1: cache built system prompt. Invalidate when memory or lessons change.
        # Env snapshot and repo map are static after construction.
        self._system_prompt_cache: str | None = None
        self._last_prompt_key: tuple[str, str] = ("", "")
        # W1.1 (SOTA v2): byte-stable prompt layout — session-static prefix
        # (instructions + env + repo map) first, volatile suffix (memory +
        # lessons) after a CACHE_BREAK sentinel that llm.py splits on.
        self._stable_prompt = stable_prompt
        # W3.1/W3.2: track failure patterns from the last completed turn.
        self.last_failure_patterns: list[str] = []
        # W1.2 (SOTA v2): append-only thread invariant. Fail-open in
        # production, fail-hard under NORN_STRICT_INVARIANTS=1 (tests/CI).
        self._ledger: ThreadLedger | None = ThreadLedger() if thread_invariants else None
        # A1 (wave 2): persistent task scratchpad, injected each round.
        self._task_notes = task_notes

    def _build_system_prompt(self) -> str:
        """Build system prompt with optional memory + lessons injection."""
        memory_content = self.memory_store.read_memory() if self.memory_store is not None else ""
        lessons_content = self.memory_store.read_lessons() if self.memory_store is not None else ""
        prompt_key = (memory_content, lessons_content)

        if self._system_prompt_cache is not None and prompt_key == self._last_prompt_key:
            return self._system_prompt_cache

        if self._stable_prompt:
            # Byte-stable layout: everything fixed for the session first, then
            # the volatile memory/lessons suffix behind a CACHE_BREAK sentinel
            # so llm.py can cache the static prefix independently.
            from norn.core.prompts import CACHE_BREAK

            prompt = self.system_prompt
            if self._env_snapshot:
                prompt += "\n\n" + self._env_snapshot
            if self._repo_map:
                prompt += "\n\n" + self._repo_map
            volatile = ""
            if memory_content.strip():
                volatile += "\n\n## Persistent Memory\n\n" + memory_content
            if lessons_content.strip():
                volatile += "\n\n## Learned Lessons\n\n" + lessons_content
            if volatile:
                prompt += CACHE_BREAK + volatile.lstrip("\n")
        else:
            prompt = self.system_prompt
            if memory_content.strip():
                prompt += "\n\n## Persistent Memory\n\n" + memory_content
            if lessons_content.strip():
                prompt += "\n\n## Learned Lessons\n\n" + lessons_content
            if self._env_snapshot:
                prompt += "\n\n" + self._env_snapshot
            if self._repo_map:
                prompt += "\n\n" + self._repo_map

        self._system_prompt_cache = prompt
        self._last_prompt_key = prompt_key
        return prompt

    @staticmethod
    def _detect_failure_patterns(messages: list[Message], final_content: str | None) -> list[str]:
        """Detect failure patterns in a completed turn. Returns pattern name list."""
        import json as _json

        patterns: list[str] = []

        if final_content == "[Max tool rounds reached]":
            patterns.append("max_rounds")

        # Tool loop: same tool + args called 3+ times
        call_counts: dict[str, int] = {}
        for msg in messages:
            if msg.role == Role.ASSISTANT and msg.tool_calls:
                for tc in msg.tool_calls:
                    key = f"{tc.name}:{_json.dumps(tc.arguments, sort_keys=True)}"
                    call_counts[key] = call_counts.get(key, 0) + 1
        for key, count in call_counts.items():
            if count >= 3:
                patterns.append(f"tool_loop:{key.split(':', 1)[0]}")

        # Error flood: 3+ consecutive tool result errors
        consecutive = 0
        max_consecutive = 0
        for msg in messages:
            if msg.role == Role.TOOL:
                content = msg.content or ""
                is_error = any(
                    content.startswith(prefix)
                    for prefix in ("Exit code", "Error:", "Unknown tool:", "Permission denied", "Timeout:")
                )
                consecutive = consecutive + 1 if is_error else 0
                max_consecutive = max(max_consecutive, consecutive)
        if max_consecutive >= 3:
            patterns.append("error_flood")

        return patterns

    def _check_invariants(self, outbound: list[Message]) -> None:
        """Validate the append-only thread invariant before an LLM dispatch.

        Fail-open in production (log ``invariant.violation``, keep going);
        strict mode raises ``ThreadInvariantError`` from the ledger itself.
        """
        if self._ledger is None:
            return
        try:
            self._ledger.extend(self.history)
            violations = self._ledger.verify(self.history)
            violations += self._ledger.verify_derivation(outbound, self.history)
        except ThreadInvariantError:
            raise
        except Exception:
            return  # observability of the check itself is fail-open
        if violations:
            with contextlib.suppress(Exception):
                _log.error(EventName.INVARIANT_VIOLATION, violations=violations)

    def _inject_task_notes(self, messages: list[Message]) -> list[Message]:
        """Insert the task scratchpad as a SYSTEM message after the prefix.

        A1 (wave 2): rendered from a file, so it survives sliding-window
        eviction. SYSTEM role → exempt from ThreadLedger.verify_derivation;
        placed after messages[0] so it never touches the prompt-cache
        breakpoint. No-op when notes are disabled or empty.
        """
        if self._task_notes is None:
            return messages
        rendered = self._task_notes.render()
        if not rendered:
            return messages
        note = Message(role=Role.SYSTEM, content=rendered)
        insert_at = 1 if messages and messages[0].role == Role.SYSTEM else 0
        return [*messages[:insert_at], note, *messages[insert_at:]]

    def _post_turn_hook(self, messages: list[Message], final_content: str | None) -> None:
        """Log failure patterns after a turn completes. Fail-open."""
        self.last_failure_patterns = self._detect_failure_patterns(messages, final_content)
        if self.last_failure_patterns and self.memory_store is not None:
            from datetime import datetime, timezone

            date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
            entry = f"[{ts}] Failure patterns: {', '.join(self.last_failure_patterns)}"
            with contextlib.suppress(Exception):
                self.memory_store.append_daily(date_str, entry)

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

        if self._context_manager is not None:
            messages = await self._context_manager.build_messages(
                self._build_system_prompt(), self.history
            )
        else:
            messages = [
                Message(role=Role.SYSTEM, content=self._build_system_prompt()),
                *self.history,
            ]

        for _round in range(self._max_tool_rounds):
            self._turn_budget.reset()

            # Re-render task notes each round so mid-run updates are seen.
            call_messages = self._inject_task_notes(messages)
            self._check_invariants(call_messages)
            response = await self.llm.complete(
                messages=call_messages,
                tools=self._get_tools_for_turn(call_messages),
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
                self._post_turn_hook(messages, response.content)
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
                content = self._turn_budget.allocate(raw_content, self._max_tool_result_chars)
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
        self._post_turn_hook(messages, final.content)
        return final

    async def run_stream(self, user_input: str) -> AsyncIterator[AgentEvent]:
        """Run one turn of the agent loop, yielding events for real-time rendering.

        Mirrors ``_run_impl`` logic but yields ``AgentEvent`` objects
        progressively so the CLI can render text as it arrives and show
        tool progress in real time.
        """
        self.user_message_count += 1
        self.history.append(Message(role=Role.USER, content=user_input))

        if self._context_manager is not None:
            messages: list[Message] = await self._context_manager.build_messages(
                self._build_system_prompt(), self.history
            )
        else:
            messages: list[Message] = [
                Message(role=Role.SYSTEM, content=self._build_system_prompt()),
                *self.history,
            ]

        for _round in range(self._max_tool_rounds):
            self._turn_budget.reset()
            # --- Stream from LLM ---
            accumulated_content = ""
            accumulated_tool_calls: list[ToolCall] = []
            final_usage = None

            # Re-render task notes each round so mid-run updates are seen.
            call_messages = self._inject_task_notes(messages)
            self._check_invariants(call_messages)
            async for chunk in self.llm.stream(
                messages=call_messages,
                tools=self._get_tools_for_turn(call_messages),
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
                self._post_turn_hook(messages, accumulated_content)
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
                content = self._turn_budget.allocate(raw_content, self._max_tool_result_chars)

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
        self._post_turn_hook(messages, "[Max tool rounds reached]")
        yield AgentEvent(type=EventType.DONE)

    @staticmethod
    def _extract_verdict(text: str) -> str:
        """Return 'pass' / 'fail' / 'unknown' from a self-verification reply.

        Uses the LAST standalone PASS/FAIL token so intermediate mentions
        ("...or FAIL") in the agent's reasoning don't override the final verdict.
        """
        import re

        matches = re.findall(r"\b(PASS|FAIL)\b", text or "")
        if not matches:
            return "unknown"
        return matches[-1].lower()

    async def run_verified(
        self,
        user_input: str,
        *,
        max_verify_rounds: int = 2,
        verify_prompt: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        """Run the task, then self-verify up to ``max_verify_rounds`` times.

        After the main run, injects a self-verification turn (same conversation,
        so history/cache carry over). If the agent's verdict is not PASS, it has
        already been asked to fix problems with tools during that same turn;
        another verify turn re-checks. Stops early on PASS or when rounds are
        exhausted. Falls back to a plain single run when ``max_verify_rounds`` is
        0, preserving existing behaviour.
        """
        from norn.core.prompts import SELF_VERIFY_PROMPT

        prompt = verify_prompt or SELF_VERIFY_PROMPT

        async for event in self.run_stream(user_input):
            yield event

        for _round in range(max_verify_rounds):
            verdict_text: list[str] = []
            async for event in self.run_stream(prompt):
                if event.type == EventType.TEXT_DELTA and event.content:
                    verdict_text.append(event.content)
                yield event
            if self._extract_verdict("".join(verdict_text)) == "pass":
                break

    def _get_tools_for_turn(self, messages: list[Message]) -> list[dict] | None:
        """Get tool schemas for this turn, optionally filtered by selector."""
        all_schemas = self.registry.get_schemas(minify=self._minify_tool_schemas)
        if not all_schemas:
            return None
        if self._tool_selector is None:
            return all_schemas

        # Get the last user message for keyword matching
        last_user_msg = ""
        recent_context = ""
        for msg in reversed(messages):
            if msg.role == Role.USER and msg.content and not last_user_msg:
                last_user_msg = msg.content
            elif msg.role == Role.ASSISTANT and msg.content and not recent_context:
                recent_context = msg.content
            if last_user_msg and recent_context:
                break

        available_names = [s["name"] for s in all_schemas]
        selected_names = self._tool_selector.select(
            last_user_msg, available_names, recent_context=recent_context
        )
        selected_set = set(selected_names)
        filtered = [s for s in all_schemas if s["name"] in selected_set]
        return filtered or None

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
