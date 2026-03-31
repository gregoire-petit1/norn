"""Tests for the feature flag registry."""

import os
from unittest.mock import patch

import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry


@pytest.fixture
def registry():
    return FeatureFlagRegistry(
        flags={
            "dream_system": FeatureFlag(
                name="dream_system", default=False, description="Memory consolidation"
            ),
            "coordinator": FeatureFlag(
                name="coordinator", default=False, description="Multi-agent mode"
            ),
            "ml_tools": FeatureFlag(
                name="ml_tools", default=True, description="MLOps-specific tools"
            ),
        }
    )


class TestDefaults:
    def test_default_false(self, registry):
        assert registry.is_enabled("dream_system") is False

    def test_default_true(self, registry):
        assert registry.is_enabled("ml_tools") is True

    def test_unknown_flag_returns_false(self, registry):
        assert registry.is_enabled("nonexistent") is False


class TestConfigOverride:
    def test_config_overrides_default(self, registry):
        registry.apply_config({"dream_system": True})
        assert registry.is_enabled("dream_system") is True

    def test_config_does_not_affect_unspecified(self, registry):
        registry.apply_config({"dream_system": True})
        assert registry.is_enabled("coordinator") is False  # Still default

    def test_config_can_disable_default_true(self, registry):
        registry.apply_config({"ml_tools": False})
        assert registry.is_enabled("ml_tools") is False


class TestEnvOverride:
    def test_env_overrides_default(self, registry):
        with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": "true"}):
            assert registry.is_enabled("dream_system") is True

    def test_env_overrides_config(self, registry):
        registry.apply_config({"dream_system": True})
        with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": "false"}):
            assert registry.is_enabled("dream_system") is False

    def test_env_true_variants(self, registry):
        for value in ["true", "1", "yes", "True", "TRUE", "YES"]:
            with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": value}):
                assert registry.is_enabled("dream_system") is True, f"Failed for: {value}"

    def test_env_false_variants(self, registry):
        for value in ["false", "0", "no", "False", "FALSE", "NO"]:
            with patch.dict(os.environ, {"NORN_FLAG_ML_TOOLS": value}):
                assert registry.is_enabled("ml_tools") is False, f"Failed for: {value}"


class TestListFlags:
    def test_list_all(self, registry):
        flags = registry.list_flags()
        assert len(flags) == 3
        names = {f.name for f in flags}
        assert names == {"dream_system", "coordinator", "ml_tools"}

    def test_list_shows_resolved_values(self, registry):
        registry.apply_config({"dream_system": True})
        flags = registry.list_flags()
        dream = next(f for f in flags if f.name == "dream_system")
        assert dream.default is False  # Original default
