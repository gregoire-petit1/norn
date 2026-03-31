"""Model Card generator tool for Norn."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class ModelCardInput(BaseModel):
    """Input for Model Card generation."""

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
    """Generate a Model Card markdown file documenting an ML model."""

    name = "model_card"
    description = "Generate a Model Card markdown document for an ML model."
    risk_level = RiskLevel.MEDIUM
    input_model = ModelCardInput

    async def execute(self, input: ModelCardInput, ctx: ToolContext) -> ToolResult:
        try:
            content = self._build_model_card(input)

            path = Path(input.output_path)
            if not path.is_absolute():
                path = Path(ctx.cwd) / path

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

            return ToolResult(output=content)
        except Exception as e:
            return ToolResult(error=f"Failed to generate model card: {e}")

    def _build_model_card(self, input: ModelCardInput) -> str:
        sections: list[str] = []

        # Model Details
        sections.append(f"# Model Card: {input.model_name}\n")
        details: list[str] = [f"- **Model Type**: {input.model_type}"]
        if input.version:
            details.append(f"- **Version**: {input.version}")
        if input.author:
            details.append(f"- **Author**: {input.author}")
        if input.license:
            details.append(f"- **License**: {input.license}")
        if input.framework:
            details.append(f"- **Framework**: {input.framework}")
        sections.append("## Model Details\n")
        sections.append("\n".join(details))

        # Description
        sections.append("\n## Description\n")
        sections.append(input.description)

        # Intended Use
        if input.intended_use:
            sections.append("\n## Intended Use\n")
            sections.append(input.intended_use)

        # Training Data
        if input.datasets:
            sections.append("\n## Training Data\n")
            sections.append("\n".join(f"- {ds}" for ds in input.datasets))

        # Evaluation Metrics
        if input.metrics:
            sections.append("\n## Evaluation Metrics\n")
            sections.append("| Metric | Value |")
            sections.append("|--------|-------|")
            for metric, value in input.metrics.items():
                formatted = f"{value:.4f}" if isinstance(value, float) else str(value)
                sections.append(f"| {metric} | {formatted} |")

        # Training Details
        if input.training_details:
            sections.append("\n## Training Details\n")
            sections.append(input.training_details)

        # Limitations
        if input.limitations:
            sections.append("\n## Limitations\n")
            sections.append("\n".join(f"- {lim}" for lim in input.limitations))

        # Ethical Considerations
        if input.ethical_considerations:
            sections.append("\n## Ethical Considerations\n")
            sections.append(input.ethical_considerations)

        return "\n".join(sections) + "\n"
