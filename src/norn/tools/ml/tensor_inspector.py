"""Tensor inspector tool for examining .pt/.pth checkpoint files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class TensorInspectorInput(BaseModel):
    """Input for tensor inspection."""

    file_path: str
    max_entries: int = 100


class TensorInspectorTool:
    """Inspect PyTorch .pt/.pth files to show tensor shapes, dtypes, and parameter counts."""

    name = "tensor_inspector"
    description = (
        "Inspect PyTorch .pt/.pth checkpoint files. Shows tensor names, shapes, "
        "dtypes, parameter counts, and file size. Useful for understanding model "
        "architecture and debugging shape mismatches."
    )
    risk_level = RiskLevel.LOW
    input_model = TensorInspectorInput

    async def execute(self, input: TensorInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import torch
        except ImportError:
            return ToolResult(error="PyTorch is not installed. Install it with: pip install torch")

        path = Path(input.file_path)
        if not path.is_absolute():
            path = Path(ctx.cwd) / path

        if not path.exists():
            return ToolResult(error=f"File not found: {path}")

        try:
            data = torch.load(str(path), map_location="cpu", weights_only=False)
        except Exception as e:
            return ToolResult(error=f"Failed to load file: {e}")

        file_size = path.stat().st_size

        # Handle different data types
        if isinstance(data, torch.Tensor):
            return self._format_single_tensor(data, file_size)
        elif isinstance(data, dict):
            return self._format_dict(data, input.max_entries, file_size)
        else:
            return ToolResult(
                output=(
                    f"File: {path.name} ({_format_size(file_size)})\n"
                    f"Type: {type(data).__name__}\n"
                    f"Unable to extract tensor information from this type."
                )
            )

    def _format_single_tensor(self, tensor: Any, file_size: int) -> ToolResult:
        """Format output for a single tensor."""
        shape = list(tensor.shape)
        params = tensor.numel()
        lines = [
            f"Single tensor: shape={shape}, dtype={tensor.dtype}, params={params:,}",
            f"File size: {_format_size(file_size)}",
        ]
        return ToolResult(output="\n".join(lines))

    def _format_dict(self, data: dict[str, Any], max_entries: int, file_size: int) -> ToolResult:
        """Format output for a dict (state_dict or checkpoint)."""
        tensors: list[tuple[str, Any]] = []
        non_tensors: list[tuple[str, Any]] = []

        _extract_tensors(data, "", tensors, non_tensors)

        lines: list[str] = []

        # Tensor table
        if tensors:
            # Header
            lines.append(f"{'Name':<50} {'Shape':<20} {'Dtype':<12} {'Params':>12}")
            lines.append("-" * 96)

            total_params = 0
            displayed = 0
            for name, tensor in tensors:
                if displayed >= max_entries:
                    remaining = len(tensors) - displayed
                    lines.append(f"... and {remaining} more tensors")
                    break
                shape = list(tensor.shape)
                params = tensor.numel()
                total_params += params
                lines.append(f"{name:<50} {str(shape):<20} {str(tensor.dtype):<12} {params:>12,}")
                displayed += 1
            else:
                # Count params for any not-yet-counted tensors (all were displayed)
                pass

            # If we broke early, still need total params from all tensors
            if displayed < len(tensors):
                total_params = sum(t.numel() for _, t in tensors)

            lines.append("-" * 96)
            lines.append(f"Total tensors: {len(tensors)}")
            lines.append(f"Total parameters: {total_params:,}")

        # Non-tensor entries
        if non_tensors:
            lines.append("")
            lines.append("Non-tensor entries:")
            for name, value in non_tensors:
                val_repr = repr(value)
                if len(val_repr) > 80:
                    val_repr = val_repr[:77] + "..."
                lines.append(f"  {name}: {type(value).__name__} = {val_repr}")

        lines.append("")
        lines.append(f"File size: {_format_size(file_size)}")

        return ToolResult(output="\n".join(lines))


def _extract_tensors(
    data: dict[str, Any],
    prefix: str,
    tensors: list[tuple[str, Any]],
    non_tensors: list[tuple[str, Any]],
) -> None:
    """Recursively extract tensors from nested dicts."""
    try:
        import torch
    except ImportError:
        return

    for key, value in data.items():
        full_key = f"{prefix}{key}" if prefix else str(key)
        if isinstance(value, torch.Tensor):
            tensors.append((full_key, value))
        elif isinstance(value, dict):
            _extract_tensors(value, f"{full_key}.", tensors, non_tensors)
        else:
            non_tensors.append((full_key, value))


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable size."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
