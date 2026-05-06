"""Configuration system for Norn."""

from __future__ import annotations

import json
import os
from enum import StrEnum
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

from norn.mcp.models import MCPConfig


class PermissionMode(StrEnum):
    INTERACTIVE = "interactive"
    AUTO = "auto"
    YOLO = "yolo"
    STRICT = "strict"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5-coder:14b"
    fallback_provider: str | None = "openrouter"
    fallback_model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    api_base: str | None = None
    # Phase 9 v2 — Workstream G. Opt-out per call site; the provider also
    # gates internally on _supports_prompt_cache(model) so this is harmless
    # on non-supporting models.
    prompt_cache: bool = True


class PermissionsConfig(BaseModel):
    mode: PermissionMode = PermissionMode.INTERACTIVE


class FlagsConfig(BaseModel):
    dream_system: bool = False
    coordinator: bool = False
    ml_tools: bool = False
    web_search: bool = False
    mcp: bool = False
    bench: bool = False


class JudgeConfig(BaseModel):
    enabled: bool = True
    model: str = "anthropic/claude-sonnet-4"
    pass_threshold: float = 7.0


class BenchConfig(BaseModel):
    tasks_dir: str = "benchmarks/tasks"
    results_dir: str = "benchmarks/results"
    reports_dir: str = "benchmarks/reports"
    default_n_runs: int = 1
    default_timeout_seconds: int = 300
    judge: JudgeConfig = JudgeConfig()
    parallel_tasks: int = 1


class MemorySystemConfig(BaseModel):
    enabled: bool = True
    memory_dir: str = "~/.norn/memory"
    dream_interval_hours: int = 24
    dream_min_sessions: int = 5
    dream_model: str | None = None  # None = use main LLM


class CoordinatorConfig(BaseModel):
    enabled: bool = False
    activation_threshold: int = 2
    max_workers_per_phase: int = 5
    worker_model: str | None = None  # None = use main LLM


class RouterTierConfig(BaseModel):
    provider: str
    model: str
    api_base: str | None = None


class RouterConfig(BaseModel):
    enabled: bool = False
    tiers: dict[str, RouterTierConfig] = Field(default_factory=dict)


class LoggingConfig(BaseModel):
    enabled: bool = True
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    output: Literal["console", "file", "both"] = "both"
    file_dir: str = "~/.norn/logs"
    include_cost: bool = True
    redact_keys: list[str] = Field(
        default_factory=lambda: ["api_key", "authorization", "token", "password"]
    )


class AgentConfig(BaseModel):
    max_tool_rounds: int = 25
    minify_tool_schemas: bool = True
    max_tool_result_chars: int = 8000
    max_turn_output_chars: int = 30000  # Per-turn budget across all tool calls
    env_bootstrap: bool = True  # Phase 10: inject environment snapshot


class ContextConfig(BaseModel):
    sliding_window: bool = False  # opt-in initially
    max_history_tokens: int = 8000
    recent_turns_keep: int = 6
    summary_max_tokens: int = 300
    summary_model: str | None = None  # None = use primary model


class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    agent: AgentConfig = AgentConfig()
    context: ContextConfig = ContextConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
    mcp: MCPConfig = MCPConfig()
    router: RouterConfig = RouterConfig()
    logging: LoggingConfig = LoggingConfig()
    bench: BenchConfig = BenchConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> NornConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load(cls) -> NornConfig:
        """Load config with fallback: project -> user -> defaults."""
        candidates = [
            Path.cwd() / ".norn" / "config.yaml",
            Path.home() / ".norn" / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml",
        ]
        for path in candidates:
            if path.exists():
                return cls.from_yaml(path)
        return cls()

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_map: dict[str, tuple[str, str]] = {
            "NORN_LLM_PROVIDER": ("llm", "provider"),
            "NORN_LLM_MODEL": ("llm", "model"),
            "NORN_PERMISSION_MODE": ("permissions", "mode"),
        }
        for env_key, (section, field) in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                section_obj = getattr(self, section)
                # Use Pydantic's field validation to ensure proper type coercion
                # (e.g., str -> PermissionMode enum)
                validated = type(section_obj).model_validate(
                    {**section_obj.model_dump(), field: value}
                )
                setattr(self, section, validated)

        # Generic JSON overrides via NORN_CONFIG_OVERRIDES
        raw = os.environ.get("NORN_CONFIG_OVERRIDES")
        if raw:
            try:
                overrides = json.loads(raw)
            except (json.JSONDecodeError, ValueError):
                return
            if isinstance(overrides, dict):
                current = self.model_dump()
                for key, val in overrides.items():
                    if key in current and isinstance(current[key], dict) and isinstance(val, dict):
                        current[key] = {**current[key], **val}
                    else:
                        current[key] = val
                merged = type(self).model_validate(current)
                for key in overrides:
                    if hasattr(merged, key):
                        setattr(self, key, getattr(merged, key))
