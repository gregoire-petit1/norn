"""Model evaluation tool — computes classification or regression metrics from CSV predictions."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class ModelEvalInput(BaseModel):
    """Input for model evaluation."""

    file_path: str
    y_true_col: str = "y_true"
    y_pred_col: str = "y_pred"
    task: str | None = None


class ModelEvalTool:
    """Evaluate model predictions from a CSV file.

    Computes classification metrics (accuracy, precision, recall, F1, confusion matrix)
    or regression metrics (MSE, RMSE, MAE, R²) depending on the task type.
    Auto-detects the task when not specified.
    """

    name = "model_eval"
    description = (
        "Evaluate model predictions from a CSV file, computing classification "
        "or regression metrics."
    )
    risk_level = RiskLevel.LOW
    input_model = ModelEvalInput

    async def execute(self, input: ModelEvalInput, ctx: ToolContext) -> ToolResult:
        # --- lazy imports for graceful degradation ---
        try:
            import polars as pl
        except ImportError:
            return ToolResult(
                error="polars is required but not installed (`pip install polars`).",
                error_type=ToolErrorType.NOT_SUPPORTED,
            )

        try:
            import sklearn.metrics  # noqa: F401
        except ImportError:
            return ToolResult(
                error="scikit-learn is required but not installed (`pip install scikit-learn`).",
                error_type=ToolErrorType.NOT_SUPPORTED,
            )

        # --- resolve & validate file ---
        path = Path(input.file_path)
        if not path.is_absolute():
            path = Path(ctx.cwd) / path

        if not path.exists():
            return ToolResult(
                error=f"File not found: {path}",
                error_type=ToolErrorType.FILE_NOT_FOUND,
            )

        # --- read CSV ---
        try:
            df = pl.read_csv(path)
        except Exception as e:
            return ToolResult(
                error=f"Failed to read CSV: {e}",
                error_type=ToolErrorType.PARSE_ERROR,
            )

        # --- validate columns ---
        if input.y_true_col not in df.columns:
            return ToolResult(
                error=f"Column '{input.y_true_col}' not found. Available columns: {df.columns}",
                error_type=ToolErrorType.INVALID_ARGUMENT,
            )
        if input.y_pred_col not in df.columns:
            return ToolResult(
                error=f"Column '{input.y_pred_col}' not found. Available columns: {df.columns}",
                error_type=ToolErrorType.INVALID_ARGUMENT,
            )

        y_true = df[input.y_true_col].to_list()
        y_pred = df[input.y_pred_col].to_list()

        # --- determine task ---
        task = input.task
        if task is None:
            task = _auto_detect_task(y_true, y_pred)

        # --- compute metrics ---
        if task == "classification":
            report = _classification_report(y_true, y_pred)
        else:
            report = _regression_report(y_true, y_pred)

        header = f"Task: {task}\n"
        return ToolResult(output=header + report)


def _auto_detect_task(y_true: list, y_pred: list) -> str:
    """Heuristic: few unique int-like values → classification, otherwise regression."""
    combined = set(y_true) | set(y_pred)
    all_int = all(isinstance(v, int) or (isinstance(v, float) and v == int(v)) for v in combined)
    if all_int and len(combined) <= 20:
        return "classification"
    return "regression"


def _classification_report(y_true: list, y_pred: list) -> str:
    """Build a classification metrics report."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    labels = sorted(set(y_true) | set(y_pred))
    is_binary = len(labels) <= 2
    average = "binary" if is_binary else "weighted"

    # Force int labels when possible so sklearn doesn't complain
    y_true_c = [int(v) for v in y_true]
    y_pred_c = [int(v) for v in y_pred]

    acc = accuracy_score(y_true_c, y_pred_c)
    prec = precision_score(y_true_c, y_pred_c, average=average, zero_division=0)
    rec = recall_score(y_true_c, y_pred_c, average=average, zero_division=0)
    f1 = f1_score(y_true_c, y_pred_c, average=average, zero_division=0)
    int_labels = [int(label) for label in labels]
    cm = confusion_matrix(y_true_c, y_pred_c, labels=int_labels)
    cls_report = classification_report(y_true_c, y_pred_c, labels=int_labels, zero_division=0)

    lines = [
        f"Accuracy:  {acc:.4f}",
        f"Precision: {prec:.4f}",
        f"Recall:    {rec:.4f}",
        f"F1:        {f1:.4f}",
        "",
        "Classification Report:",
        cls_report,
        "Confusion Matrix:",
        str(cm),
    ]
    return "\n".join(lines)


def _regression_report(y_true: list, y_pred: list) -> str:
    """Build a regression metrics report."""
    import math

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = math.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    lines = [
        f"MSE:  {mse:.6f}",
        f"RMSE: {rmse:.6f}",
        f"MAE:  {mae:.6f}",
        f"R\u00b2:   {r2:.6f}",
    ]
    return "\n".join(lines)
