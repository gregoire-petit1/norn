"""image_read — describe an image via a multimodal LLM sub-call (KIRA-style).

Terminal-bench has a class of tasks that hinge on *seeing* a file — a chess
position rendered to PNG, a plot, a scanned document. A text-only agent parses
the pixels heuristically and gets it wrong (chess-best-move, gcode-to-text).

This tool keeps the main conversation text-only: it base64-encodes the image,
makes a SEPARATE multimodal ``litellm.acompletion`` call (image + instruction),
and returns the model's textual description as the tool result. The big base64
blob never enters the main thread, so context and prompt-cache stay clean.

Requires a vision-capable model (Claude, GPT-4o, Gemini …); the model id is
resolved from config at registry-build time.
"""

from __future__ import annotations

import base64
from pathlib import Path

from pydantic import BaseModel, Field

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult

_MIME_BY_EXT = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

# Guard against sending an oversized payload the provider will 400 on.
_MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB


class ImageReadInput(BaseModel):
    """Input for image_read."""

    file_path: str = Field(description="Path to the image file (png/jpg/gif/webp)")
    instruction: str = Field(
        description="What to look for or extract from the image, e.g. "
        "'transcribe the chess position in FEN' or 'read all visible text'"
    )


class ImageReadTool:
    """Analyze an image file with a vision model and return a text description."""

    name = "image_read"
    description = (
        "Look at an image file (png/jpg/gif/webp) and return a text description. "
        "Use this for any task that requires SEEING a file — a rendered board, a "
        "chart, a scanned document — instead of guessing from bytes. Provide a "
        "specific instruction for what to extract."
    )
    risk_level = RiskLevel.LOW
    input_model = ImageReadInput

    def __init__(self, model: str, api_base: str | None = None) -> None:
        self._model = model
        self._api_base = api_base

    async def execute(self, input: ImageReadInput, ctx: ToolContext) -> ToolResult:
        path = Path(input.file_path)
        if not path.is_absolute():
            path = Path(ctx.cwd) / path

        if not path.exists():
            return ToolResult(
                error=f"Image not found: {input.file_path}",
                error_type=ToolErrorType.FILE_NOT_FOUND.value,
            )

        mime = _MIME_BY_EXT.get(path.suffix.lower())
        if mime is None:
            return ToolResult(
                error=(
                    f"Unsupported image format '{path.suffix}'. Convert to PNG first "
                    f"(e.g. `convert {path.name} out.png`), then image_read the PNG."
                ),
                error_type=ToolErrorType.NOT_SUPPORTED.value,
            )

        try:
            raw = path.read_bytes()
        except OSError as exc:
            return ToolResult(error=str(exc), error_type=ToolErrorType.EXECUTION_ERROR.value)

        if len(raw) > _MAX_IMAGE_BYTES:
            return ToolResult(
                error=(
                    f"Image too large ({len(raw)} bytes > {_MAX_IMAGE_BYTES}). "
                    "Downscale it first (e.g. `convert in.png -resize 50% out.png`)."
                ),
                error_type=ToolErrorType.RESOURCE_EXHAUSTED.value,
            )

        b64 = base64.b64encode(raw).decode("ascii")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": input.instruction},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ]

        try:
            import litellm

            kwargs: dict = {
                "model": self._model,
                "messages": messages,
                "temperature": 0.0,
                "max_tokens": 2048,
                # Providers that don't accept a param (e.g. reasoning_effort) drop
                # it rather than 400 the whole vision call.
                "drop_params": True,
            }
            if self._api_base:
                kwargs["api_base"] = self._api_base
            response = await litellm.acompletion(**kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(
                error=f"Vision call failed: {exc}",
                error_type=ToolErrorType.EXECUTION_ERROR.value,
            )

        try:
            text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            text = ""
        if not text.strip():
            return ToolResult(
                error="Vision model returned no description.",
                error_type=ToolErrorType.EXECUTION_ERROR.value,
            )

        return ToolResult(output=text)
