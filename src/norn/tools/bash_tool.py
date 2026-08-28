"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from typing import TYPE_CHECKING

from pydantic import BaseModel

from norn.observability import EventName, get_logger
from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult

if TYPE_CHECKING:
    from norn.permissions.checker import PromptFn

_log = get_logger(__name__)


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str
    # Default sized for compute-heavy steps (builds, solver/engine runs). The
    # old 120s cut off legitimate long analyses mid-task; the model can still
    # raise this per call for known-slow commands.
    timeout: int = 300
    # SOTA v2 (workstream C): per-call confinement override when the sandbox
    # is enabled ("read-only" | "workspace-write" | "danger-full-access").
    # Requesting danger-full-access triggers a user-approval escalation.
    sandbox_policy: str | None = None


class BashTool:
    """Execute a shell command."""

    name = "bash"
    description = (
        "Execute a bash command in the shell. Default timeout 300s; pass a larger "
        "`timeout` (seconds) for known-slow commands like builds or long computations."
    )
    risk_level = RiskLevel.HIGH
    input_model = BashInput

    def __init__(
        self,
        *,
        sandbox_enabled: bool = False,
        default_policy: str = "workspace-write",
        allow_network: bool = False,
        extra_write_paths: list[str] | None = None,
        escalation_fn: PromptFn | None = None,
    ) -> None:
        self._sandbox_enabled = sandbox_enabled
        self._default_policy = default_policy
        self._allow_network = allow_network
        self._extra_write_paths = extra_write_paths or []
        self._escalation_fn = escalation_fn

    def _emit_sandbox_decision(self, **fields: object) -> None:
        with contextlib.suppress(Exception):
            _log.info(EventName.SANDBOX_DECISION, **fields)

    async def _resolve_sandbox_argv(
        self, input: BashInput, ctx: ToolContext, wrapped: str
    ) -> list[str] | ToolResult | None:
        """Resolve confinement for this call.

        Returns an argv (confined execution), None (unconfined execution —
        sandbox off, or escalation approved), or a ToolResult refusal.
        Fail-closed: confinement requested + mechanism unavailable → refusal.
        """
        if not self._sandbox_enabled:
            return None

        from norn.sandbox import SandboxPolicy, build_argv, is_available

        try:
            policy = SandboxPolicy(input.sandbox_policy or self._default_policy)
        except ValueError:
            return ToolResult(
                error=f"Unknown sandbox_policy '{input.sandbox_policy}'. "
                f"Valid: {', '.join(p.value for p in SandboxPolicy)}",
                error_type=ToolErrorType.INVALID_ARGUMENT.value,
            )

        if policy == SandboxPolicy.DANGER_FULL_ACCESS:
            # Escalation is a user interaction, never an automatic grant.
            granted = False
            if self._escalation_fn is not None:
                from norn.permissions.models import PermissionRequest

                request = PermissionRequest(
                    tool_name=self.name,
                    risk_level=RiskLevel.HIGH.value,
                    arguments={"command": input.command, "sandbox_policy": policy.value},
                )
                with contextlib.suppress(Exception):
                    granted = await self._escalation_fn(
                        request, "bash requests UNSANDBOXED execution (danger-full-access)"
                    )
            self._emit_sandbox_decision(
                policy=policy.value, mechanism=None, available=True, escalated=True, granted=granted
            )
            if not granted:
                return ToolResult(
                    error="Unsandboxed execution (danger-full-access) denied"
                    + ("" if self._escalation_fn else " — no escalation handler"),
                    error_type=ToolErrorType.PERMISSION_DENIED.value,
                )
            return None  # approved: run unconfined

        if not is_available():
            # Fail-closed invariant: NEVER silently fall back to unconfined.
            self._emit_sandbox_decision(
                policy=policy.value, mechanism="seatbelt", available=False,
                escalated=False, granted=False,
            )
            return ToolResult(
                error="Sandbox requested but unavailable on this platform "
                "(seatbelt requires macOS + /usr/bin/sandbox-exec). "
                "Refusing unsandboxed execution.",
                error_type=ToolErrorType.SANDBOX_UNAVAILABLE.value,
            )

        self._emit_sandbox_decision(
            policy=policy.value, mechanism="seatbelt", available=True,
            escalated=False, granted=True,
        )
        return build_argv(
            policy,
            ctx.cwd,
            wrapped,
            allow_network=self._allow_network,
            extra_write_paths=self._extra_write_paths,
        )

    async def execute(self, input: BashInput, ctx: ToolContext) -> ToolResult:
        seq = uuid.uuid4().hex[:8]
        marker = f"__NORN_{seq}__"
        # Preserve exit code before printing the marker so callers can distinguish
        # success from failure. printf avoids locale-dependent echo -e behaviour.
        wrapped = (
            f"{input.command}\n"
            f"__norn_rc=$?\n"
            f"printf '\\n{marker}\\n'\n"
            f"exit $__norn_rc"
        )
        sandbox_argv = await self._resolve_sandbox_argv(input, ctx, wrapped)
        if isinstance(sandbox_argv, ToolResult):
            return sandbox_argv
        try:
            if sandbox_argv is not None:
                # Confined: same script, same /bin/sh -c semantics, inside
                # seatbelt. A policy deny surfaces as rc != 0 with
                # "Operation not permitted" on stderr via the normal path.
                process = await asyncio.create_subprocess_exec(
                    *sandbox_argv,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=ctx.cwd,
                )
            else:
                process = await asyncio.create_subprocess_shell(
                    wrapped,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=ctx.cwd,
                )
        except Exception as e:
            return ToolResult(error=str(e), error_type=ToolErrorType.EXECUTION_ERROR.value)

        # Drain stderr concurrently to prevent pipe deadlock on commands with
        # large stderr output (compiler warnings, install logs, etc.).
        stderr_task = asyncio.create_task(process.stderr.read())  # type: ignore[union-attr]

        stdout_parts: list[str] = []
        loop = asyncio.get_running_loop()
        deadline = loop.time() + input.timeout

        try:
            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    raise asyncio.TimeoutError
                line_bytes = await asyncio.wait_for(
                    process.stdout.readline(),  # type: ignore[union-attr]
                    timeout=remaining,
                )
                if not line_bytes:
                    # EOF: process exited before printing marker (e.g. killed, crash)
                    break
                line = line_bytes.decode("utf-8", errors="replace")
                if marker in line:
                    break
                stdout_parts.append(line)
        except asyncio.TimeoutError:
            # Reap the subprocess transport WITHIN the running loop. Skipping
            # `await process.wait()` here leaves the transport unreaped; its
            # __del__ then fires after the event loop has closed and raises
            # "RuntimeError: Event loop is closed", which crashed whole headless
            # runs mid-task (e.g. a stockfish analysis that overran the timeout).
            with contextlib.suppress(ProcessLookupError):
                process.kill()
            stderr_task.cancel()
            # CancelledError subclasses BaseException, not Exception — suppress
            # it explicitly so awaiting the just-cancelled task doesn't re-raise.
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await stderr_task
            with contextlib.suppress(Exception):
                await process.wait()
            return ToolResult(
                error=f"Timeout: command exceeded {input.timeout}s",
                error_type=ToolErrorType.TIMEOUT.value,
            )

        # Drain remaining stderr (fast: process is done by the time marker is printed).
        try:
            stderr_bytes = await asyncio.wait_for(stderr_task, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            stderr_bytes = b""
        await process.wait()

        output = "".join(stdout_parts)
        errors = stderr_bytes.decode("utf-8", errors="replace")

        if process.returncode != 0:
            return ToolResult(
                error=f"Exit code {process.returncode}\n{errors or output}",
                error_type=ToolErrorType.EXECUTION_ERROR.value,
            )

        combined = output
        if errors:
            combined += f"\nSTDERR:\n{errors}"
        return ToolResult(output=combined)
