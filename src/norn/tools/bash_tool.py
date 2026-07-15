"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio
import contextlib
import uuid

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str
    # Default sized for compute-heavy steps (builds, solver/engine runs). The
    # old 120s cut off legitimate long analyses mid-task; the model can still
    # raise this per call for known-slow commands.
    timeout: int = 300


class BashTool:
    """Execute a shell command."""

    name = "bash"
    description = (
        "Execute a bash command in the shell. Default timeout 300s; pass a larger "
        "`timeout` (seconds) for known-slow commands like builds or long computations."
    )
    risk_level = RiskLevel.HIGH
    input_model = BashInput

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
        try:
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
