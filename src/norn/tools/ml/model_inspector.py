"""Model inspector tool for PyTorch nn.Module analysis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class ModelInspectorInput(BaseModel):
    """Input for model inspection."""

    file_path: str
    class_name: str
    constructor_args: dict[str, Any] | None = None


class ModelInspectorTool:
    """Inspect a PyTorch nn.Module: architecture, parameter counts, and memory estimate."""

    name = "model_inspector"
    description = (
        "Inspect a PyTorch nn.Module from a Python file. "
        "Reports layer structure, parameter counts, and estimated memory usage."
    )
    risk_level = RiskLevel.LOW
    input_model = ModelInspectorInput

    async def execute(self, input: ModelInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import torch  # noqa: F401
            import torch.nn as nn
        except ImportError:
            return ToolResult(error="PyTorch is not installed. Install with: pip install torch")

        filepath = Path(input.file_path)
        if not filepath.is_absolute():
            filepath = Path(ctx.cwd) / filepath

        if not filepath.exists():
            return ToolResult(error=f"File not found: {filepath}")

        # Dynamically load the module from the file path.
        module_name = f"_norn_inspector_{filepath.stem}"
        spec = importlib.util.spec_from_file_location(module_name, str(filepath))
        if spec is None or spec.loader is None:
            return ToolResult(error=f"Cannot load module from: {filepath}")

        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            cls = getattr(module, input.class_name, None)
            if cls is None:
                return ToolResult(error=f"Class '{input.class_name}' not found in {filepath.name}")

            # Instantiate the model.
            args = input.constructor_args or {}
            try:
                model = cls(**args)
            except Exception as e:
                return ToolResult(error=f"Failed to instantiate {input.class_name}: {e}")

            if not isinstance(model, nn.Module):
                return ToolResult(error=f"'{input.class_name}' is not an nn.Module subclass")

            # Collect layer info.
            lines: list[str] = []
            lines.append(f"Model: {input.class_name}")
            lines.append("=" * 60)

            total_params = 0
            trainable_params = 0

            for name, mod in model.named_modules():
                if name == "":
                    continue  # Skip the root module itself.
                params = sum(p.numel() for p in mod.parameters(recurse=False))
                train = sum(p.numel() for p in mod.parameters(recurse=False) if p.requires_grad)
                total_params += params
                trainable_params += train
                lines.append(f"  {name}: {mod.__class__.__name__} — {params:,} params")

            lines.append("=" * 60)
            lines.append(f"Total parameters: {total_params:,}")
            lines.append(f"Trainable parameters: {trainable_params:,}")

            # Memory estimate (float32 = 4 bytes per param).
            memory_bytes = total_params * 4
            if memory_bytes < 1024:
                mem_str = f"{memory_bytes} B"
            elif memory_bytes < 1024 * 1024:
                mem_str = f"{memory_bytes / 1024:.1f} KB"
            elif memory_bytes < 1024 * 1024 * 1024:
                mem_str = f"{memory_bytes / (1024 * 1024):.1f} MB"
            else:
                mem_str = f"{memory_bytes / (1024 * 1024 * 1024):.2f} GB"

            lines.append(f"Estimated memory (f32): {mem_str}")

            return ToolResult(output="\n".join(lines))

        except Exception as e:
            return ToolResult(error=f"Error inspecting model: {e}")
        finally:
            sys.modules.pop(module_name, None)
