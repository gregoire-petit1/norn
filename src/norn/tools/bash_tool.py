"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio
import uuid

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class BashInput(BaseModel):
    """Input for bash command execution."""

    command: str
    timeout: int = 120


class BashTool:
    """Execute a shell command."""

    name = "bash"
    description = "Execute a bash command in the shell."
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
            process.kill()
            stderr_task.cancel()
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
