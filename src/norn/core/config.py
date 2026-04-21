"""Configuration system for Norn."""

from __future__ import annotations

import os
from enum import StrEnum
from pathlib import Path

import yaml
from pydantic import BaseModel

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


class PermissionsConfig(BaseModel):
    mode: PermissionMode = PermissionMode.INTERACTIVE


class FlagsConfig(BaseModel):
    dream_system: bool = False
    coordinator: bool = False
    ml_tools: bool = False
    web_search: bool = False
    mcp: bool = False


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
    tiers: dict[str, RouterTierConfig] = {}


class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
    mcp: MCPConfig = MCPConfig()
    router: RouterConfig = RouterConfig()

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
