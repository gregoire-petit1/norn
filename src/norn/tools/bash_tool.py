"""Bash tool for shell command execution."""

from __future__ import annotations

import asyncio

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


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
        try:
            process = await asyncio.create_subprocess_shell(
                input.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ctx.cwd,
            )
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=input.timeout)

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return ToolResult(error=f"Exit code {process.returncode}\n{errors or output}")

            combined = output
            if errors:
                combined += f"\nSTDERR:\n{errors}"
            return ToolResult(output=combined)

        except TimeoutError:
            process.kill()
            return ToolResult(error=f"Timeout: command exceeded {input.timeout}s")
        except Exception as e:
            return ToolResult(error=str(e))
