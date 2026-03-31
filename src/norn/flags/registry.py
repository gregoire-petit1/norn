"""Feature flag registry with priority resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass

_TRUTHY = {"true", "1", "yes"}
_FALSY = {"false", "0", "no"}


@dataclass
class FeatureFlag:
    """A single feature flag definition."""

    name: str
    default: bool
    description: str = ""


class FeatureFlagRegistry:
    """Registry for feature flags with priority: env > config > default."""

    def __init__(self, flags: dict[str, FeatureFlag] | None = None) -> None:
        self._flags: dict[str, FeatureFlag] = flags or {}
        self._config_overrides: dict[str, bool] = {}

    def register(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self._flags[flag.name] = flag

    def apply_config(self, config: dict[str, bool]) -> None:
        """Apply config-level overrides."""
        self._config_overrides.update(config)

    def is_enabled(self, name: str) -> bool:
        """Resolve a flag value with priority: env > config > default."""
        flag = self._flags.get(name)
        if flag is None:
            return False

        # Priority 1: Environment variable
        env_key = f"NORN_FLAG_{name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value.lower() in _TRUTHY

        # Priority 2: Config override
        if name in self._config_overrides:
            return self._config_overrides[name]

        # Priority 3: Default
        return flag.default

    def list_flags(self) -> list[FeatureFlag]:
        """List all registered flags."""
        return list(self._flags.values())
