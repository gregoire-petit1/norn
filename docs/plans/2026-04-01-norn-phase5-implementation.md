# Norn Phase 5: ML-Specific Tools Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build 5 ML-specific tools that give the Norn agent native understanding of PyTorch models, datasets (via Polars), sklearn evaluation, model cards, and tensor/checkpoint inspection. All tools are gated behind the existing `ml_tools` feature flag.

**Architecture:** Each tool follows the existing `Tool` protocol (name, description, risk_level, input_model, async execute). ML dependencies (torch, polars, scikit-learn) are **optional** — declared in `[project.optional-dependencies] ml`. Tools gracefully return `ToolResult(error="...")` when their dependencies are missing, so the agent loop never crashes. All 5 tools are registered with `feature_flag="ml_tools"` in `_build_registry()`. No new configs needed — the existing `FlagsConfig.ml_tools` boolean is sufficient.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, pytest, ruff. New optional deps: `torch>=2.0`, `polars>=1.0`, `scikit-learn>=1.5`.

**Tools:**

| # | Tool | Risk | Purpose |
|---|------|------|---------|
| 1 | `model_inspector` | LOW | Inspect PyTorch nn.Module: layer summary, param counts, memory estimate |
| 2 | `tensor_inspector` | LOW | Inspect .pt/.safetensors checkpoint files: tensor names, shapes, dtypes, size |
| 3 | `dataset_inspector` | LOW | Inspect CSV/Parquet datasets via Polars: shape, dtypes, nulls, stats, sample |
| 4 | `model_eval` | LOW | Evaluate sklearn-compatible predictions: classification + regression metrics |
| 5 | `model_card` | MEDIUM | Generate a Model Card markdown file from structured metadata |

---

## Task 0: Optional ML Dependencies and Directory Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `src/norn/tools/ml/__init__.py`
- Create: `tests/test_tools/test_ml/__init__.py`

**Step 1: Create directories**

```bash
mkdir -p src/norn/tools/ml tests/test_tools/test_ml
touch src/norn/tools/ml/__init__.py tests/test_tools/test_ml/__init__.py
```

**Step 2: Add optional ML dependencies to pyproject.toml**

Add to `pyproject.toml` after the `dev` optional-dependencies block:

```toml
ml = [
    "torch>=2.0",
    "polars>=1.0",
    "scikit-learn>=1.5",
]
```

**Step 3: Install ML dependencies**

```bash
uv pip install -e ".[ml]"
```

**Step 4: Verify imports**

```bash
uv run python -c "import torch; import polars; import sklearn; print('OK')"
```

Expected: `OK`

**Step 5: Run existing tests (no regression)**

Run: `uv run pytest tests/ -v --tb=short -q`
Expected: 275 passed, 0 failed

**Step 6: Commit**

```bash
git add pyproject.toml src/norn/tools/ml/__init__.py tests/test_tools/test_ml/__init__.py
git commit -m "chore: add ml optional dependencies and tools/ml package structure"
```

---

## Task 1: Model Inspector Tool

Inspects a PyTorch `nn.Module` defined in a Python file. Loads the module class, instantiates it, and returns a summary: layer names, parameter counts (total/trainable), estimated memory, and dtype info.

**Files:**
- Create: `src/norn/tools/ml/model_inspector.py`
- Create: `tests/test_tools/test_ml/test_model_inspector.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_model_inspector.py
"""Tests for ModelInspectorTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool


@pytest.fixture
def tool():
    return ModelInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "model_inspector"
    assert tool.risk_level == RiskLevel.LOW
    assert tool.description


@pytest.mark.asyncio
async def test_inspect_model_file(tool, tmp_path):
    """Inspect a simple PyTorch model defined in a file."""
    model_file = tmp_path / "my_model.py"
    model_file.write_text(
        "import torch.nn as nn\n\n"
        "class SimpleNet(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc1 = nn.Linear(10, 20)\n"
        "        self.fc2 = nn.Linear(20, 5)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc2(self.fc1(x))\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="SimpleNet"), ctx
    )
    assert result.is_error is False
    assert "SimpleNet" in result.output
    assert "fc1" in result.output
    assert "fc2" in result.output
    # Should report parameter counts
    assert "325" in result.output  # 10*20 + 20 + 20*5 + 5 = 325


@pytest.mark.asyncio
async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path="/nonexistent/model.py", class_name="Foo"), ctx
    )
    assert result.is_error is True
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_inspect_missing_class(tool, tmp_path):
    model_file = tmp_path / "empty_model.py"
    model_file.write_text("import torch.nn as nn\n\nclass Foo(nn.Module):\n    pass\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="Bar"), ctx
    )
    assert result.is_error is True
    assert "Bar" in result.error


@pytest.mark.asyncio
async def test_inspect_with_constructor_args(tool, tmp_path):
    """Model that requires constructor arguments."""
    model_file = tmp_path / "param_model.py"
    model_file.write_text(
        "import torch.nn as nn\n\n"
        "class ParamNet(nn.Module):\n"
        "    def __init__(self, in_features: int, out_features: int):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(in_features, out_features)\n"
        "\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelInspectorInput(
            file_path=str(model_file),
            class_name="ParamNet",
            constructor_args={"in_features": 8, "out_features": 4},
        ),
        ctx,
    )
    assert result.is_error is False
    assert "ParamNet" in result.output
    assert "fc" in result.output
    # 8*4 + 4 = 36 params
    assert "36" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_ml/test_model_inspector.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.tools.ml.model_inspector'`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/ml/model_inspector.py
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
    """Inspect a PyTorch nn.Module: layers, parameters, memory estimate."""

    name = "model_inspector"
    description = (
        "Inspect a PyTorch model defined in a Python file. "
        "Returns layer summary, parameter counts, and memory estimate."
    )
    risk_level = RiskLevel.LOW
    input_model = ModelInspectorInput

    async def execute(self, input: ModelInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import torch.nn as nn
        except ImportError:
            return ToolResult(error="PyTorch is not installed. Install with: uv pip install 'norn[ml]'")

        file_path = Path(input.file_path)
        if not file_path.is_absolute():
            file_path = Path(ctx.cwd) / file_path

        if not file_path.exists():
            return ToolResult(error=f"File not found: {file_path}")

        try:
            # Dynamically load the module from file
            spec = importlib.util.spec_from_file_location("_norn_inspect_module", str(file_path))
            if spec is None or spec.loader is None:
                return ToolResult(error=f"Cannot load module from: {file_path}")

            module = importlib.util.module_from_spec(spec)
            sys.modules["_norn_inspect_module"] = module
            spec.loader.exec_module(module)

            # Find the class
            cls = getattr(module, input.class_name, None)
            if cls is None:
                return ToolResult(error=f"Class '{input.class_name}' not found in {file_path}")

            if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
                return ToolResult(error=f"'{input.class_name}' is not a nn.Module subclass")

            # Instantiate the model
            args = input.constructor_args or {}
            model = cls(**args)

            # Build summary
            lines: list[str] = []
            lines.append(f"# Model: {input.class_name}")
            lines.append("")

            # Layer summary
            lines.append("## Layers")
            lines.append(f"{'Name':<30} {'Type':<25} {'Params':>12}")
            lines.append("-" * 69)

            total_params = 0
            trainable_params = 0

            for name, param_module in model.named_modules():
                if name == "":
                    continue
                module_params = sum(p.numel() for p in param_module.parameters(recurse=False))
                module_trainable = sum(
                    p.numel() for p in param_module.parameters(recurse=False) if p.requires_grad
                )
                if module_params > 0:
                    lines.append(
                        f"{name:<30} {type(param_module).__name__:<25} {module_params:>12,}"
                    )
                total_params += module_params
                trainable_params += module_trainable

            # If named_modules gave nothing useful, fall back to parameters
            if total_params == 0:
                total_params = sum(p.numel() for p in model.parameters())
                trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

            lines.append("")
            lines.append("## Summary")
            lines.append(f"- Total parameters: {total_params:,}")
            lines.append(f"- Trainable parameters: {trainable_params:,}")
            lines.append(f"- Non-trainable parameters: {total_params - trainable_params:,}")

            # Memory estimate (assuming float32 by default)
            bytes_per_param = 4  # float32
            first_param = next(model.parameters(), None)
            if first_param is not None:
                bytes_per_param = first_param.element_size()
            memory_bytes = total_params * bytes_per_param
            if memory_bytes < 1024 * 1024:
                mem_str = f"{memory_bytes / 1024:.1f} KB"
            elif memory_bytes < 1024 * 1024 * 1024:
                mem_str = f"{memory_bytes / (1024 * 1024):.1f} MB"
            else:
                mem_str = f"{memory_bytes / (1024 * 1024 * 1024):.2f} GB"
            lines.append(f"- Estimated memory (weights): {mem_str}")

            return ToolResult(output="\n".join(lines))

        except Exception as e:
            return ToolResult(error=f"Inspection failed: {e}")
        finally:
            # Clean up the temp module
            sys.modules.pop("_norn_inspect_module", None)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_ml/test_model_inspector.py -v`
Expected: 5 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/ml/model_inspector.py`

**Step 6: Commit**

```bash
git add src/norn/tools/ml/model_inspector.py tests/test_tools/test_ml/test_model_inspector.py
git commit -m "feat(ml): add model_inspector tool for PyTorch nn.Module analysis"
```

---

## Task 2: Tensor Inspector Tool

Inspects `.pt` checkpoint files (and plain tensor files) without loading the full model. Reports tensor names, shapes, dtypes, and total size.

**Files:**
- Create: `src/norn/tools/ml/tensor_inspector.py`
- Create: `tests/test_tools/test_ml/test_tensor_inspector.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_tensor_inspector.py
"""Tests for TensorInspectorTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool


@pytest.fixture
def tool():
    return TensorInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "tensor_inspector"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_inspect_state_dict(tool, tmp_path):
    """Inspect a saved state_dict checkpoint."""
    import torch
    import torch.nn as nn

    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 5))
    checkpoint_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint_path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(checkpoint_path)), ctx)
    assert result.is_error is False
    assert "0.weight" in result.output
    assert "0.bias" in result.output
    assert "2.weight" in result.output
    assert "[10, 20]" in result.output or "[20, 10]" in result.output


@pytest.mark.asyncio
async def test_inspect_single_tensor(tool, tmp_path):
    """Inspect a file containing a single tensor."""
    import torch

    tensor = torch.randn(3, 4, 5)
    tensor_path = tmp_path / "tensor.pt"
    torch.save(tensor, tensor_path)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(tensor_path)), ctx)
    assert result.is_error is False
    assert "[3, 4, 5]" in result.output


@pytest.mark.asyncio
async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path="/no/such/file.pt"), ctx)
    assert result.is_error is True
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_inspect_dict_checkpoint(tool, tmp_path):
    """Inspect a checkpoint dict with model + optimizer state."""
    import torch
    import torch.nn as nn

    model = nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    ckpt_path = tmp_path / "full_ckpt.pt"
    torch.save(
        {"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "epoch": 10},
        ckpt_path,
    )

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(TensorInspectorInput(file_path=str(ckpt_path)), ctx)
    assert result.is_error is False
    # Should report the top-level keys
    assert "model_state_dict" in result.output
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_ml/test_tensor_inspector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/ml/tensor_inspector.py
"""Tensor inspector tool for PyTorch checkpoint files."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class TensorInspectorInput(BaseModel):
    """Input for tensor/checkpoint inspection."""

    file_path: str
    max_entries: int = 100


class TensorInspectorTool:
    """Inspect PyTorch .pt checkpoint files: tensor names, shapes, dtypes, sizes."""

    name = "tensor_inspector"
    description = (
        "Inspect a PyTorch .pt checkpoint file. "
        "Reports tensor names, shapes, dtypes, and total size."
    )
    risk_level = RiskLevel.LOW
    input_model = TensorInspectorInput

    async def execute(self, input: TensorInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import torch
        except ImportError:
            return ToolResult(error="PyTorch is not installed. Install with: uv pip install 'norn[ml]'")

        file_path = Path(input.file_path)
        if not file_path.is_absolute():
            file_path = Path(ctx.cwd) / file_path

        if not file_path.exists():
            return ToolResult(error=f"File not found: {file_path}")

        try:
            data = torch.load(file_path, map_location="cpu", weights_only=False)
        except Exception as e:
            return ToolResult(error=f"Failed to load checkpoint: {e}")

        lines: list[str] = []
        lines.append(f"# Checkpoint: {file_path.name}")
        lines.append(f"- File size: {_format_size(file_path.stat().st_size)}")
        lines.append("")

        if isinstance(data, torch.Tensor):
            lines.append("## Single Tensor")
            lines.append(_format_tensor("<root>", data))
            lines.append("")
            lines.append(f"- Total elements: {data.numel():,}")
        elif isinstance(data, dict):
            # Check if it's a state_dict (all values are tensors) or a full checkpoint
            tensor_entries: dict[str, torch.Tensor] = {}
            other_entries: dict[str, str] = {}
            _extract_tensors(data, "", tensor_entries, other_entries)

            if tensor_entries:
                lines.append("## Tensors")
                lines.append(f"{'Name':<45} {'Shape':<20} {'Dtype':<12} {'Params':>12}")
                lines.append("-" * 91)

                total_params = 0
                shown = 0
                for name, tensor in tensor_entries.items():
                    if shown >= input.max_entries:
                        lines.append(f"  ... ({len(tensor_entries) - shown} more entries)")
                        break
                    lines.append(_format_tensor(name, tensor))
                    total_params += tensor.numel()
                    shown += 1

                lines.append("")
                lines.append(f"- Total tensors: {len(tensor_entries):,}")
                lines.append(f"- Total parameters: {total_params:,}")

            if other_entries:
                lines.append("")
                lines.append("## Non-Tensor Entries")
                for key, desc in other_entries.items():
                    lines.append(f"- `{key}`: {desc}")
        else:
            lines.append(f"- Data type: {type(data).__name__}")

        return ToolResult(output="\n".join(lines))


def _format_tensor(name: str, tensor: "torch.Tensor") -> str:
    """Format a single tensor entry."""
    shape = str(list(tensor.shape))
    return f"{name:<45} {shape:<20} {str(tensor.dtype):<12} {tensor.numel():>12,}"


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def _extract_tensors(
    data: dict,
    prefix: str,
    tensors: dict,
    others: dict,
) -> None:
    """Recursively extract tensors and non-tensor entries from a dict."""
    import torch

    for key, value in data.items():
        full_key = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
        if isinstance(value, torch.Tensor):
            tensors[full_key] = value
        elif isinstance(value, dict):
            _extract_tensors(value, full_key, tensors, others)
        else:
            others[full_key] = f"{type(value).__name__}: {str(value)[:80]}"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_ml/test_tensor_inspector.py -v`
Expected: 5 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/ml/tensor_inspector.py`

**Step 6: Commit**

```bash
git add src/norn/tools/ml/tensor_inspector.py tests/test_tools/test_ml/test_tensor_inspector.py
git commit -m "feat(ml): add tensor_inspector tool for PyTorch checkpoint analysis"
```

---

## Task 3: Dataset Inspector Tool

Inspects CSV and Parquet datasets using Polars: shape, column types, null counts, basic statistics, and optional sample rows.

**Files:**
- Create: `src/norn/tools/ml/dataset_inspector.py`
- Create: `tests/test_tools/test_ml/test_dataset_inspector.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_dataset_inspector.py
"""Tests for DatasetInspectorTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.dataset_inspector import DatasetInspectorInput, DatasetInspectorTool


@pytest.fixture
def tool():
    return DatasetInspectorTool()


def test_tool_metadata(tool):
    assert tool.name == "dataset_inspector"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_inspect_csv(tool, tmp_path):
    """Inspect a simple CSV dataset."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("name,age,score\nAlice,30,95.5\nBob,25,87.0\nCharlie,,91.2\n")

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(csv_file)), ctx)
    assert result.is_error is False
    assert "3" in result.output  # 3 rows
    assert "name" in result.output
    assert "age" in result.output
    assert "score" in result.output


@pytest.mark.asyncio
async def test_inspect_parquet(tool, tmp_path):
    """Inspect a Parquet file."""
    import polars as pl

    df = pl.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.0, 6.0], "label": ["a", "b", "a"]})
    parquet_file = tmp_path / "data.parquet"
    df.write_parquet(parquet_file)

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(parquet_file)), ctx)
    assert result.is_error is False
    assert "3" in result.output  # 3 rows
    assert "x" in result.output
    assert "y" in result.output
    assert "label" in result.output


@pytest.mark.asyncio
async def test_inspect_with_sample(tool, tmp_path):
    """Request sample rows."""
    csv_file = tmp_path / "data.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n5,6\n7,8\n9,10\n")

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        DatasetInspectorInput(file_path=str(csv_file), show_sample=True, sample_rows=2), ctx
    )
    assert result.is_error is False
    assert "Sample" in result.output


@pytest.mark.asyncio
async def test_inspect_null_stats(tool, tmp_path):
    """Reports null/missing counts."""
    csv_file = tmp_path / "nulls.csv"
    csv_file.write_text("a,b,c\n1,,x\n2,3,\n,4,y\n")

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(csv_file)), ctx)
    assert result.is_error is False
    # Should report null counts
    assert "null" in result.output.lower() or "missing" in result.output.lower()


@pytest.mark.asyncio
async def test_inspect_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path="/no/data.csv"), ctx)
    assert result.is_error is True
    assert "not found" in result.error.lower()


@pytest.mark.asyncio
async def test_inspect_unsupported_format(tool, tmp_path):
    txt_file = tmp_path / "data.txt"
    txt_file.write_text("hello world")

    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(DatasetInspectorInput(file_path=str(txt_file)), ctx)
    assert result.is_error is True
    assert "unsupported" in result.error.lower()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_ml/test_dataset_inspector.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/ml/dataset_inspector.py
"""Dataset inspector tool using Polars for CSV/Parquet analysis."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult

_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


class DatasetInspectorInput(BaseModel):
    """Input for dataset inspection."""

    file_path: str
    show_sample: bool = False
    sample_rows: int = 5


class DatasetInspectorTool:
    """Inspect CSV/Parquet datasets: shape, types, nulls, statistics, sample."""

    name = "dataset_inspector"
    description = (
        "Inspect a CSV or Parquet dataset using Polars. "
        "Returns shape, column types, null counts, basic statistics, and optional sample rows."
    )
    risk_level = RiskLevel.LOW
    input_model = DatasetInspectorInput

    async def execute(self, input: DatasetInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import polars as pl
        except ImportError:
            return ToolResult(
                error="Polars is not installed. Install with: uv pip install 'norn[ml]'"
            )

        file_path = Path(input.file_path)
        if not file_path.is_absolute():
            file_path = Path(ctx.cwd) / file_path

        if not file_path.exists():
            return ToolResult(error=f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            return ToolResult(
                error=f"Unsupported format: '{suffix}'. Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}"
            )

        try:
            if suffix == ".csv":
                df = pl.read_csv(file_path, infer_schema_length=1000)
            else:
                df = pl.read_parquet(file_path)
        except Exception as e:
            return ToolResult(error=f"Failed to read dataset: {e}")

        lines: list[str] = []
        lines.append(f"# Dataset: {file_path.name}")
        lines.append(f"- Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
        lines.append(f"- File size: {_format_size(file_path.stat().st_size)}")
        lines.append("")

        # Column info
        lines.append("## Columns")
        lines.append(f"{'Column':<30} {'Type':<15} {'Nulls':>8} {'Null%':>8}")
        lines.append("-" * 63)

        null_counts = df.null_count()
        for col in df.columns:
            dtype = str(df[col].dtype)
            n_nulls = null_counts[col][0]
            null_pct = (n_nulls / df.shape[0] * 100) if df.shape[0] > 0 else 0.0
            lines.append(f"{col:<30} {dtype:<15} {n_nulls:>8,} {null_pct:>7.1f}%")

        lines.append("")

        # Numeric statistics
        numeric_cols = [col for col in df.columns if df[col].dtype.is_numeric()]
        if numeric_cols:
            lines.append("## Numeric Statistics")
            try:
                desc = df.select(numeric_cols).describe()
                lines.append(str(desc))
            except Exception:
                lines.append("(Statistics unavailable)")
            lines.append("")

        # Sample rows
        if input.show_sample:
            n = min(input.sample_rows, df.shape[0])
            lines.append(f"## Sample ({n} rows)")
            lines.append(str(df.head(n)))

        return ToolResult(output="\n".join(lines))


def _format_size(size_bytes: int) -> str:
    """Format file size in human-readable form."""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    if size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    if size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_ml/test_dataset_inspector.py -v`
Expected: 7 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/ml/dataset_inspector.py`

**Step 6: Commit**

```bash
git add src/norn/tools/ml/dataset_inspector.py tests/test_tools/test_ml/test_dataset_inspector.py
git commit -m "feat(ml): add dataset_inspector tool for CSV/Parquet analysis via Polars"
```

---

## Task 4: Model Evaluation Tool

Evaluates predictions from a CSV file using sklearn metrics. Supports classification (accuracy, precision, recall, F1, confusion matrix) and regression (MSE, RMSE, MAE, R²).

**Files:**
- Create: `src/norn/tools/ml/model_eval.py`
- Create: `tests/test_tools/test_ml/test_model_eval.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_model_eval.py
"""Tests for ModelEvalTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool


@pytest.fixture
def tool():
    return ModelEvalTool()


def test_tool_metadata(tool):
    assert tool.name == "model_eval"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_classification_eval(tool, tmp_path):
    """Evaluate classification predictions."""
    csv_file = tmp_path / "preds.csv"
    csv_file.write_text(
        "y_true,y_pred\n"
        "0,0\n1,1\n0,0\n1,0\n1,1\n0,1\n1,1\n0,0\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(
            file_path=str(csv_file),
            y_true_col="y_true",
            y_pred_col="y_pred",
            task="classification",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "accuracy" in result.output.lower()
    assert "precision" in result.output.lower()
    assert "recall" in result.output.lower()
    assert "f1" in result.output.lower()


@pytest.mark.asyncio
async def test_regression_eval(tool, tmp_path):
    """Evaluate regression predictions."""
    csv_file = tmp_path / "reg_preds.csv"
    csv_file.write_text(
        "actual,predicted\n"
        "1.0,1.1\n2.0,2.2\n3.0,2.8\n4.0,4.1\n5.0,5.3\n"
    )
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(
            file_path=str(csv_file),
            y_true_col="actual",
            y_pred_col="predicted",
            task="regression",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "mse" in result.output.lower()
    assert "mae" in result.output.lower()
    assert "r²" in result.output.lower() or "r2" in result.output.lower()


@pytest.mark.asyncio
async def test_auto_detect_classification(tool, tmp_path):
    """Auto-detect task type as classification for discrete values."""
    csv_file = tmp_path / "preds.csv"
    csv_file.write_text("y_true,y_pred\n0,0\n1,1\n0,1\n1,1\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(file_path=str(csv_file), y_true_col="y_true", y_pred_col="y_pred"),
        ctx,
    )
    assert result.is_error is False
    assert "classification" in result.output.lower()


@pytest.mark.asyncio
async def test_multiclass_eval(tool, tmp_path):
    """Evaluate multiclass classification."""
    csv_file = tmp_path / "multi.csv"
    csv_file.write_text("y_true,y_pred\n0,0\n1,1\n2,2\n0,1\n1,2\n2,0\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(
            file_path=str(csv_file),
            y_true_col="y_true",
            y_pred_col="y_pred",
            task="classification",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "accuracy" in result.output.lower()


@pytest.mark.asyncio
async def test_missing_column(tool, tmp_path):
    csv_file = tmp_path / "preds.csv"
    csv_file.write_text("a,b\n1,2\n3,4\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(file_path=str(csv_file), y_true_col="y_true", y_pred_col="y_pred"),
        ctx,
    )
    assert result.is_error is True
    assert "y_true" in result.error


@pytest.mark.asyncio
async def test_nonexistent_file(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelEvalInput(file_path="/no/preds.csv", y_true_col="y", y_pred_col="p"), ctx
    )
    assert result.is_error is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_ml/test_model_eval.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/ml/model_eval.py
"""Model evaluation tool using scikit-learn metrics."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class ModelEvalInput(BaseModel):
    """Input for model evaluation."""

    file_path: str
    y_true_col: str = "y_true"
    y_pred_col: str = "y_pred"
    task: str | None = None  # "classification" | "regression" | None (auto-detect)


class ModelEvalTool:
    """Evaluate model predictions: classification or regression metrics from a CSV."""

    name = "model_eval"
    description = (
        "Evaluate model predictions from a CSV file. "
        "Supports classification (accuracy, precision, recall, F1, confusion matrix) "
        "and regression (MSE, RMSE, MAE, R²). Auto-detects task type if not specified."
    )
    risk_level = RiskLevel.LOW
    input_model = ModelEvalInput

    async def execute(self, input: ModelEvalInput, ctx: ToolContext) -> ToolResult:
        try:
            import polars as pl
        except ImportError:
            return ToolResult(
                error="Polars is not installed. Install with: uv pip install 'norn[ml]'"
            )

        try:
            from sklearn import metrics as skmetrics
        except ImportError:
            return ToolResult(
                error="scikit-learn is not installed. Install with: uv pip install 'norn[ml]'"
            )

        file_path = Path(input.file_path)
        if not file_path.is_absolute():
            file_path = Path(ctx.cwd) / file_path

        if not file_path.exists():
            return ToolResult(error=f"File not found: {file_path}")

        try:
            df = pl.read_csv(file_path)
        except Exception as e:
            return ToolResult(error=f"Failed to read CSV: {e}")

        if input.y_true_col not in df.columns:
            return ToolResult(
                error=f"Column '{input.y_true_col}' not found. Available: {df.columns}"
            )
        if input.y_pred_col not in df.columns:
            return ToolResult(
                error=f"Column '{input.y_pred_col}' not found. Available: {df.columns}"
            )

        y_true = df[input.y_true_col].to_list()
        y_pred = df[input.y_pred_col].to_list()

        # Auto-detect task type
        task = input.task
        if task is None:
            unique_true = set(y_true)
            unique_pred = set(y_pred)
            # Heuristic: if few unique values and all are ints, it's classification
            if len(unique_true) <= 20 and all(
                isinstance(v, (int, float)) and float(v) == int(v)
                for v in unique_true | unique_pred
                if v is not None
            ):
                task = "classification"
            else:
                task = "regression"

        lines: list[str] = []
        lines.append(f"# Model Evaluation ({task})")
        lines.append(f"- Samples: {len(y_true):,}")
        lines.append("")

        if task == "classification":
            lines.extend(_classification_report(y_true, y_pred, skmetrics))
        else:
            lines.extend(_regression_report(y_true, y_pred, skmetrics))

        return ToolResult(output="\n".join(lines))


def _classification_report(y_true: list, y_pred: list, skmetrics: object) -> list[str]:
    """Generate classification metrics."""
    import numpy as np

    lines: list[str] = []
    lines.append("## Metrics")

    accuracy = skmetrics.accuracy_score(y_true, y_pred)
    lines.append(f"- Accuracy: {accuracy:.4f}")

    # Determine averaging strategy
    unique_labels = sorted(set(y_true) | set(y_pred))
    avg = "binary" if len(unique_labels) <= 2 else "weighted"

    precision = skmetrics.precision_score(y_true, y_pred, average=avg, zero_division=0)
    recall = skmetrics.recall_score(y_true, y_pred, average=avg, zero_division=0)
    f1 = skmetrics.f1_score(y_true, y_pred, average=avg, zero_division=0)

    lines.append(f"- Precision: {precision:.4f}")
    lines.append(f"- Recall: {recall:.4f}")
    lines.append(f"- F1-score: {f1:.4f}")
    lines.append("")

    # Classification report
    lines.append("## Classification Report")
    lines.append("```")
    lines.append(skmetrics.classification_report(y_true, y_pred, zero_division=0))
    lines.append("```")

    # Confusion matrix
    lines.append("")
    lines.append("## Confusion Matrix")
    cm = skmetrics.confusion_matrix(y_true, y_pred)
    lines.append("```")
    lines.append(str(np.array(cm)))
    lines.append("```")

    return lines


def _regression_report(y_true: list, y_pred: list, skmetrics: object) -> list[str]:
    """Generate regression metrics."""
    import math

    lines: list[str] = []
    lines.append("## Metrics")

    mse = skmetrics.mean_squared_error(y_true, y_pred)
    mae = skmetrics.mean_absolute_error(y_true, y_pred)
    r2 = skmetrics.r2_score(y_true, y_pred)
    rmse = math.sqrt(mse)

    lines.append(f"- MSE: {mse:.6f}")
    lines.append(f"- RMSE: {rmse:.6f}")
    lines.append(f"- MAE: {mae:.6f}")
    lines.append(f"- R²: {r2:.6f}")

    return lines
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_ml/test_model_eval.py -v`
Expected: 7 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/ml/model_eval.py`

**Step 6: Commit**

```bash
git add src/norn/tools/ml/model_eval.py tests/test_tools/test_ml/test_model_eval.py
git commit -m "feat(ml): add model_eval tool for sklearn classification/regression metrics"
```

---

## Task 5: Model Card Generator Tool

Generates a standardized Model Card markdown file from structured metadata. Follows the "Model Cards for Model Reporting" format (Mitchell et al., 2019). The tool writes a `.md` file and returns its content.

**Files:**
- Create: `src/norn/tools/ml/model_card.py`
- Create: `tests/test_tools/test_ml/test_model_card.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_model_card.py
"""Tests for ModelCardTool."""

import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.ml.model_card import ModelCardInput, ModelCardTool


@pytest.fixture
def tool():
    return ModelCardTool()


def test_tool_metadata(tool):
    assert tool.name == "model_card"
    assert tool.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_generate_basic_card(tool, tmp_path):
    """Generate a minimal model card."""
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=str(tmp_path / "MODEL_CARD.md"),
            model_name="MyClassifier",
            model_type="classification",
            description="A simple binary classifier for sentiment.",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "MyClassifier" in result.output

    # File should be written
    card_path = tmp_path / "MODEL_CARD.md"
    assert card_path.exists()
    content = card_path.read_text()
    assert "MyClassifier" in content
    assert "classification" in content.lower()


@pytest.mark.asyncio
async def test_generate_full_card(tool, tmp_path):
    """Generate a full model card with all optional fields."""
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=str(tmp_path / "CARD.md"),
            model_name="ResNet50-Finetuned",
            model_type="image_classification",
            description="Fine-tuned ResNet50 on medical imaging.",
            version="2.1.0",
            author="ML Team",
            license="Apache-2.0",
            framework="PyTorch",
            datasets=["ImageNet", "ChestX-ray14"],
            metrics={"accuracy": 0.94, "f1": 0.92, "auc": 0.97},
            intended_use="Classify chest X-rays as normal/abnormal.",
            limitations=["Not validated on pediatric images", "Requires 512x512 input"],
            ethical_considerations="Model may underperform on underrepresented demographics.",
            training_details="Fine-tuned for 50 epochs, lr=1e-4, AdamW, batch_size=32.",
        ),
        ctx,
    )
    assert result.is_error is False
    content = (tmp_path / "CARD.md").read_text()
    assert "ResNet50-Finetuned" in content
    assert "Apache-2.0" in content
    assert "ImageNet" in content
    assert "0.94" in content
    assert "pediatric" in content
    assert "Fine-tuned for 50 epochs" in content


@pytest.mark.asyncio
async def test_generate_card_creates_parent_dirs(tool, tmp_path):
    """Output path with non-existent parent dirs should be created."""
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path=str(tmp_path / "docs" / "cards" / "CARD.md"),
            model_name="TestModel",
            model_type="regression",
            description="A test model.",
        ),
        ctx,
    )
    assert result.is_error is False
    assert (tmp_path / "docs" / "cards" / "CARD.md").exists()


@pytest.mark.asyncio
async def test_card_relative_path(tool, tmp_path):
    """Relative output path should resolve against cwd."""
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        ModelCardInput(
            output_path="MODEL_CARD.md",
            model_name="RelModel",
            model_type="classification",
            description="Relative path test.",
        ),
        ctx,
    )
    assert result.is_error is False
    assert (tmp_path / "MODEL_CARD.md").exists()
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_ml/test_model_card.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/ml/model_card.py
"""Model card generator tool."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class ModelCardInput(BaseModel):
    """Input for model card generation."""

    output_path: str
    model_name: str
    model_type: str
    description: str
    version: str | None = None
    author: str | None = None
    license: str | None = None
    framework: str | None = None
    datasets: list[str] | None = None
    metrics: dict[str, Any] | None = None
    intended_use: str | None = None
    limitations: list[str] | None = None
    ethical_considerations: str | None = None
    training_details: str | None = None


class ModelCardTool:
    """Generate a standardized Model Card markdown file."""

    name = "model_card"
    description = (
        "Generate a Model Card markdown file from structured metadata. "
        "Follows the 'Model Cards for Model Reporting' format."
    )
    risk_level = RiskLevel.MEDIUM  # Writes a file
    input_model = ModelCardInput

    async def execute(self, input: ModelCardInput, ctx: ToolContext) -> ToolResult:
        output_path = Path(input.output_path)
        if not output_path.is_absolute():
            output_path = Path(ctx.cwd) / output_path

        try:
            content = _build_model_card(input)

            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")

            return ToolResult(output=f"Model card written to {output_path}\n\n{content}")
        except Exception as e:
            return ToolResult(error=f"Failed to generate model card: {e}")


def _build_model_card(input: ModelCardInput) -> str:
    """Build the model card markdown content."""
    lines: list[str] = []

    # Header
    lines.append(f"# Model Card: {input.model_name}")
    lines.append("")

    # Overview
    lines.append("## Model Details")
    lines.append("")
    lines.append(f"- **Name:** {input.model_name}")
    lines.append(f"- **Type:** {input.model_type}")
    if input.version:
        lines.append(f"- **Version:** {input.version}")
    if input.author:
        lines.append(f"- **Author:** {input.author}")
    if input.license:
        lines.append(f"- **License:** {input.license}")
    if input.framework:
        lines.append(f"- **Framework:** {input.framework}")
    lines.append("")

    # Description
    lines.append("## Description")
    lines.append("")
    lines.append(input.description)
    lines.append("")

    # Intended Use
    if input.intended_use:
        lines.append("## Intended Use")
        lines.append("")
        lines.append(input.intended_use)
        lines.append("")

    # Training Data
    if input.datasets:
        lines.append("## Training Data")
        lines.append("")
        for ds in input.datasets:
            lines.append(f"- {ds}")
        lines.append("")

    # Metrics
    if input.metrics:
        lines.append("## Evaluation Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for metric, value in input.metrics.items():
            if isinstance(value, float):
                lines.append(f"| {metric} | {value:.4f} |")
            else:
                lines.append(f"| {metric} | {value} |")
        lines.append("")

    # Training Details
    if input.training_details:
        lines.append("## Training Details")
        lines.append("")
        lines.append(input.training_details)
        lines.append("")

    # Limitations
    if input.limitations:
        lines.append("## Limitations")
        lines.append("")
        for limitation in input.limitations:
            lines.append(f"- {limitation}")
        lines.append("")

    # Ethical Considerations
    if input.ethical_considerations:
        lines.append("## Ethical Considerations")
        lines.append("")
        lines.append(input.ethical_considerations)
        lines.append("")

    return "\n".join(lines)
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_ml/test_model_card.py -v`
Expected: 4 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/ml/model_card.py`

**Step 6: Commit**

```bash
git add src/norn/tools/ml/model_card.py tests/test_tools/test_ml/test_model_card.py
git commit -m "feat(ml): add model_card tool for standardized Model Card generation"
```

---

## Task 6: Register ML Tools in CLI

Wire all 5 ML tools into `_build_registry()` behind the `ml_tools` feature flag. Handle the case where ML dependencies are not installed (lazy imports, skip registration).

**Files:**
- Modify: `src/norn/cli/main.py`
- Create: `tests/test_tools/test_ml/test_registration.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_ml/test_registration.py
"""Tests for ML tool registration."""

import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.registry import ToolRegistry


def _build_ml_registry(ml_enabled: bool) -> ToolRegistry:
    """Build a registry with ML tools for testing."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag("ml_tools", ml_enabled, "ML tools")}
    )
    flag_registry.apply_config({"ml_tools": ml_enabled})

    registry = ToolRegistry(flag_registry=flag_registry)

    # Import and register ML tools
    from norn.tools.ml.dataset_inspector import DatasetInspectorTool
    from norn.tools.ml.model_card import ModelCardTool
    from norn.tools.ml.model_eval import ModelEvalTool
    from norn.tools.ml.model_inspector import ModelInspectorTool
    from norn.tools.ml.tensor_inspector import TensorInspectorTool

    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")

    return registry


def test_ml_tools_enabled():
    """When ml_tools flag is on, all 5 ML tools are listed."""
    registry = _build_ml_registry(ml_enabled=True)
    names = {t.name for t in registry.list_tools()}
    assert "model_inspector" in names
    assert "tensor_inspector" in names
    assert "dataset_inspector" in names
    assert "model_eval" in names
    assert "model_card" in names


def test_ml_tools_disabled():
    """When ml_tools flag is off, no ML tools are listed."""
    registry = _build_ml_registry(ml_enabled=False)
    names = {t.name for t in registry.list_tools()}
    assert "model_inspector" not in names
    assert "tensor_inspector" not in names
    assert "dataset_inspector" not in names
    assert "model_eval" not in names
    assert "model_card" not in names


def test_ml_tools_schemas_respect_flag():
    """Schemas should only include enabled tools."""
    registry_on = _build_ml_registry(ml_enabled=True)
    registry_off = _build_ml_registry(ml_enabled=False)

    schemas_on = registry_on.get_schemas()
    schemas_off = registry_off.get_schemas()

    ml_names_on = {s["name"] for s in schemas_on}
    ml_names_off = {s["name"] for s in schemas_off}

    assert "model_inspector" in ml_names_on
    assert "model_inspector" not in ml_names_off
```

**Step 2: Run tests to verify they fail (or pass — these test pure registration)**

Run: `uv run pytest tests/test_tools/test_ml/test_registration.py -v`
Expected: 3 PASSED (these should pass since the tools already exist)

**Step 3: Modify CLI to register ML tools**

In `src/norn/cli/main.py`, add ML tool imports and registration in `_build_registry()`.

After the existing tool imports (around line 37), add:

```python
from norn.tools.ml.model_inspector import ModelInspectorTool
from norn.tools.ml.tensor_inspector import TensorInspectorTool
from norn.tools.ml.dataset_inspector import DatasetInspectorTool
from norn.tools.ml.model_eval import ModelEvalTool
from norn.tools.ml.model_card import ModelCardTool
```

In `_build_registry()`, after the existing `registry.register(GrepTool())` line, add:

```python
    # ML tools (gated behind ml_tools feature flag)
    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")
```

**Step 4: Run registration tests**

Run: `uv run pytest tests/test_tools/test_ml/test_registration.py -v`
Expected: 3 PASSED

**Step 5: Run ruff on CLI**

Run: `uv run ruff check src/norn/cli/main.py`

**Step 6: Commit**

```bash
git add src/norn/cli/main.py tests/test_tools/test_ml/test_registration.py
git commit -m "feat(ml): register all ML tools in CLI behind ml_tools feature flag"
```

---

## Task 7: Update Default Config

Enable `ml_tools` by default in `configs/default.yaml` (since the tools are now implemented), and add a config show line for ML tools in the CLI `config` command.

**Files:**
- Modify: `configs/default.yaml`
- Modify: `src/norn/cli/main.py`

**Step 1: Update default.yaml**

Change `ml_tools: false` to `ml_tools: true` and update the comment:

```yaml
flags:
  dream_system: false   # Phase 3
  coordinator: false    # Phase 4
  ml_tools: true        # Phase 5: PyTorch/Polars/sklearn tools
```

**Step 2: Add ML tools status to CLI config command**

In `src/norn/cli/main.py`, in the `config()` command, after the coordinator line, add:

```python
    console.print(f"  ML tools:     {'enabled' if cfg.flags.ml_tools else 'disabled'}")
```

**Step 3: Run all existing tests (no regression)**

Run: `uv run pytest tests/ -v --tb=short -q`
Expected: All pass (275 + new tests)

**Step 4: Commit**

```bash
git add configs/default.yaml src/norn/cli/main.py
git commit -m "feat(ml): enable ml_tools by default and show status in config command"
```

---

## Task 8: Integration Tests

End-to-end tests that verify ML tools work together with the agent loop (mocked LLM), permission system, and feature flags.

**Files:**
- Create: `tests/test_tools/test_ml/test_ml_integration.py`

**Step 1: Write the integration tests**

```python
# tests/test_tools/test_ml/test_ml_integration.py
"""Integration tests for ML tools with agent infrastructure."""

import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.base import ToolContext
from norn.tools.ml.dataset_inspector import DatasetInspectorInput, DatasetInspectorTool
from norn.tools.ml.model_card import ModelCardInput, ModelCardTool
from norn.tools.ml.model_eval import ModelEvalInput, ModelEvalTool
from norn.tools.ml.model_inspector import ModelInspectorInput, ModelInspectorTool
from norn.tools.ml.tensor_inspector import TensorInspectorInput, TensorInspectorTool
from norn.tools.registry import ToolRegistry


@pytest.fixture
def full_ml_registry():
    """Build a full registry with ML tools enabled."""
    flags = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag("ml_tools", True, "ML tools")}
    )
    flags.apply_config({"ml_tools": True})
    registry = ToolRegistry(flag_registry=flags)
    registry.register(ModelInspectorTool(), feature_flag="ml_tools")
    registry.register(TensorInspectorTool(), feature_flag="ml_tools")
    registry.register(DatasetInspectorTool(), feature_flag="ml_tools")
    registry.register(ModelEvalTool(), feature_flag="ml_tools")
    registry.register(ModelCardTool(), feature_flag="ml_tools")
    return registry


def test_all_ml_tools_have_schemas(full_ml_registry):
    """All ML tools should produce valid JSON schemas."""
    schemas = full_ml_registry.get_schemas()
    names = {s["name"] for s in schemas}
    assert names == {"model_inspector", "tensor_inspector", "dataset_inspector", "model_eval", "model_card"}
    for schema in schemas:
        assert "parameters" in schema
        assert "description" in schema


def test_ml_tools_retrieved_by_name(full_ml_registry):
    """Each ML tool can be retrieved by name."""
    for name in ["model_inspector", "tensor_inspector", "dataset_inspector", "model_eval", "model_card"]:
        tool = full_ml_registry.get(name)
        assert tool is not None
        assert tool.name == name


@pytest.mark.asyncio
async def test_ml_pipeline_inspect_and_evaluate(tmp_path):
    """Simulate: inspect model -> generate predictions -> evaluate -> write card."""
    ctx = ToolContext(cwd=str(tmp_path))

    # Step 1: Inspect model
    model_file = tmp_path / "net.py"
    model_file.write_text(
        "import torch.nn as nn\n\n"
        "class Net(nn.Module):\n"
        "    def __init__(self):\n"
        "        super().__init__()\n"
        "        self.fc = nn.Linear(4, 2)\n"
        "    def forward(self, x):\n"
        "        return self.fc(x)\n"
    )
    inspector = ModelInspectorTool()
    result = await inspector.execute(
        ModelInspectorInput(file_path=str(model_file), class_name="Net"), ctx
    )
    assert result.is_error is False
    assert "Net" in result.output

    # Step 2: Create predictions CSV and evaluate
    preds = tmp_path / "preds.csv"
    preds.write_text("y_true,y_pred\n0,0\n1,1\n0,1\n1,1\n0,0\n")
    evaluator = ModelEvalTool()
    result = await evaluator.execute(
        ModelEvalInput(file_path=str(preds), y_true_col="y_true", y_pred_col="y_pred"), ctx
    )
    assert result.is_error is False
    assert "accuracy" in result.output.lower()

    # Step 3: Generate model card
    card_tool = ModelCardTool()
    result = await card_tool.execute(
        ModelCardInput(
            output_path=str(tmp_path / "MODEL_CARD.md"),
            model_name="Net",
            model_type="classification",
            description="Simple 2-class classifier.",
            metrics={"accuracy": 0.8, "f1": 0.75},
        ),
        ctx,
    )
    assert result.is_error is False
    assert (tmp_path / "MODEL_CARD.md").exists()


@pytest.mark.asyncio
async def test_dataset_then_tensor_inspection(tmp_path):
    """Inspect a dataset and a checkpoint in sequence."""
    import torch
    import torch.nn as nn

    ctx = ToolContext(cwd=str(tmp_path))

    # Create a dataset
    csv_file = tmp_path / "train.csv"
    csv_file.write_text("feature_1,feature_2,label\n1.0,2.0,0\n3.0,4.0,1\n5.0,6.0,0\n")
    ds_tool = DatasetInspectorTool()
    result = await ds_tool.execute(DatasetInspectorInput(file_path=str(csv_file)), ctx)
    assert result.is_error is False
    assert "3" in result.output  # 3 rows

    # Save and inspect a checkpoint
    model = nn.Linear(2, 1)
    ckpt_path = tmp_path / "model.pt"
    torch.save(model.state_dict(), ckpt_path)
    tensor_tool = TensorInspectorTool()
    result = await tensor_tool.execute(TensorInspectorInput(file_path=str(ckpt_path)), ctx)
    assert result.is_error is False
    assert "weight" in result.output
```

**Step 2: Run integration tests**

Run: `uv run pytest tests/test_tools/test_ml/test_ml_integration.py -v`
Expected: 4 PASSED

**Step 3: Run ruff**

Run: `uv run ruff check tests/test_tools/test_ml/test_ml_integration.py`

**Step 4: Commit**

```bash
git add tests/test_tools/test_ml/test_ml_integration.py
git commit -m "test(ml): add integration tests for ML tools pipeline"
```

---

## Task 9: Full Regression Test and Final Verification

Run all tests, ruff, verify tool count, and confirm no regressions.

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short -q`
Expected: 275 + ~33 new tests = ~308 PASSED, 0 FAILED

**Step 2: Run ruff on all ML tool files**

Run: `uv run ruff check src/norn/tools/ml/`

**Step 3: Run ruff on full codebase**

Run: `uv run ruff check src/norn/ --exclude src/norn/cli/main.py`

(Exclude main.py for the known E402 warnings.)

**Step 4: Verify tool listing**

Run: `uv run norn tools`

Expected output should include all 11 tools (6 base + 5 ML):
- bash, file_read, file_write, file_edit, glob, grep
- model_inspector, tensor_inspector, dataset_inspector, model_eval, model_card

**Step 5: Verify config shows ML tools**

Run: `uv run norn config`

Expected: `ML tools: enabled`

**Step 6: Final commit (if any ruff fixes needed)**

```bash
git add -A
git commit -m "chore: phase 5 final cleanup and verification"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 0 | Optional deps + directory setup | 0 | `pyproject.toml`, `tools/ml/__init__.py` |
| 1 | Model Inspector Tool | 5 | `tools/ml/model_inspector.py` |
| 2 | Tensor Inspector Tool | 5 | `tools/ml/tensor_inspector.py` |
| 3 | Dataset Inspector Tool | 7 | `tools/ml/dataset_inspector.py` |
| 4 | Model Eval Tool | 7 | `tools/ml/model_eval.py` |
| 5 | Model Card Tool | 4 | `tools/ml/model_card.py` |
| 6 | CLI Registration | 3 | `cli/main.py` (modify) |
| 7 | Default Config | 0 | `configs/default.yaml`, `cli/main.py` |
| 8 | Integration Tests | 4 | `test_ml_integration.py` |
| 9 | Final Verification | 0 | (none) |

**Total: ~35 new tests, 9 commits, 12 new files, 3 modified files**
**New optional dependencies: `torch>=2.0`, `polars>=1.0`, `scikit-learn>=1.5`**
