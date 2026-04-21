"""Dataset inspector tool for CSV and Parquet files."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult

_SUPPORTED_EXTENSIONS = {".csv", ".parquet", ".pq"}


class DatasetInspectorInput(BaseModel):
    """Input for dataset inspection."""

    file_path: str
    show_sample: bool = False
    sample_rows: int = 5


def _format_size(size_bytes: int) -> str:
    """Format byte count as human-readable string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}" if unit != "B" else f"{size_bytes} B"
        size_bytes /= 1024
    return f"{size_bytes:.1f} PB"


class DatasetInspectorTool:
    """Inspect CSV and Parquet datasets — shape, types, nulls, and statistics."""

    name = "dataset_inspector"
    description = (
        "Inspect a CSV or Parquet file: reports shape, column types, "
        "null counts, descriptive statistics, and optional sample rows."
    )
    risk_level = RiskLevel.LOW
    input_model = DatasetInspectorInput

    async def execute(self, input: DatasetInspectorInput, ctx: ToolContext) -> ToolResult:
        try:
            import polars as pl
        except ImportError:
            return ToolResult(
                error="polars is required for dataset inspection. "
                "Install it with: uv pip install polars",
                error_type=ToolErrorType.NOT_SUPPORTED,
            )

        path = Path(input.file_path)
        if not path.is_absolute():
            path = Path(ctx.cwd) / path

        if not path.exists():
            return ToolResult(
                error=f"File not found: {path}",
                error_type=ToolErrorType.FILE_NOT_FOUND,
            )

        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            return ToolResult(
                error=f"Unsupported format '{suffix}'. "
                f"Supported: {', '.join(sorted(_SUPPORTED_EXTENSIONS))}",
                error_type=ToolErrorType.INVALID_ARGUMENT,
            )

        try:
            df = pl.read_csv(path) if suffix == ".csv" else pl.read_parquet(path)
        except Exception as e:
            return ToolResult(
                error=f"Failed to read {path.name}: {e}",
                error_type=ToolErrorType.PARSE_ERROR,
            )

        file_size = path.stat().st_size
        rows, cols = df.shape

        # -- Build report ------------------------------------------------
        lines: list[str] = []
        lines.append(f"## Dataset: {path.name}")
        lines.append(f"- **Rows**: {rows}")
        lines.append(f"- **Columns**: {cols}")
        lines.append(f"- **File size**: {_format_size(file_size)}")
        lines.append("")

        # Column table
        lines.append("### Columns")
        lines.append(f"{'Name':<20} {'Type':<15} {'Nulls':<8} {'Null %':<8}")
        lines.append("-" * 55)
        for col_name in df.columns:
            col = df[col_name]
            null_count = col.null_count()
            null_pct = (null_count / rows * 100) if rows > 0 else 0.0
            lines.append(f"{col_name:<20} {str(col.dtype):<15} {null_count:<8} {null_pct:.1f}%")
        lines.append("")

        # Numeric statistics
        numeric_cols = [c for c in df.columns if df[c].dtype.is_numeric()]
        if numeric_cols:
            lines.append("### Statistics (numeric)")
            try:
                desc = df.select(numeric_cols).describe()
                lines.append(str(desc))
            except Exception:
                lines.append("_(statistics unavailable)_")
            lines.append("")

        # Optional sample
        if input.show_sample:
            n = min(input.sample_rows, rows)
            lines.append(f"### Sample ({n} rows)")
            lines.append(str(df.head(n)))
            lines.append("")

        return ToolResult(output="\n".join(lines))
