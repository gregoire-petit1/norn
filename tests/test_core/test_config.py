"""Tests for configuration system."""

from norn.core.config import NornConfig, PermissionMode


def test_default_config():
    config = NornConfig()
    assert config.llm.provider == "ollama"
    assert config.permissions.mode == PermissionMode.INTERACTIVE


def test_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  provider: openrouter\n  model: anthropic/claude-sonnet-4-20250514\n"
        "permissions:\n  mode: auto\n"
    )
    config = NornConfig.from_yaml(config_file)
    assert config.llm.provider == "openrouter"
    assert config.llm.model == "anthropic/claude-sonnet-4-20250514"
    assert config.permissions.mode == PermissionMode.AUTO


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("NORN_LLM_MODEL", "gpt-4o")
    config = NornConfig()
    config.apply_env_overrides()
    assert config.llm.model == "gpt-4o"


def test_config_env_override_permission_mode(monkeypatch):
    """Env override for permission mode should produce a proper PermissionMode enum."""
    monkeypatch.setenv("NORN_PERMISSION_MODE", "yolo")
    config = NornConfig()
    config.apply_env_overrides()
    assert config.permissions.mode == PermissionMode.YOLO
    # Crucially, .value must work (it failed before fix when mode was a raw str)
    assert config.permissions.mode.value == "yolo"


def test_flag_env_overrides():
    """Feature flags should be overridable via config."""
    config = NornConfig()
    assert config.flags.dream_system is False
    config.flags.dream_system = True
    assert config.flags.dream_system is True


def test_memory_config_defaults():
    """NornConfig should have memory config with defaults."""
    config = NornConfig()
    assert config.memory.enabled is True
    assert config.memory.dream_interval_hours == 24
    assert config.memory.dream_min_sessions == 5


def test_memory_config_custom():
    """Memory config should be overridable."""
    config = NornConfig(memory={"dream_interval_hours": 12, "dream_min_sessions": 3})
    assert config.memory.dream_interval_hours == 12
    assert config.memory.dream_min_sessions == 3


def test_coordinator_config_defaults():
    """NornConfig should have coordinator config with defaults."""
    config = NornConfig()
    assert config.coordinator.enabled is False
    assert config.coordinator.activation_threshold == 2
    assert config.coordinator.max_workers_per_phase == 5


def test_coordinator_config_custom():
    """Coordinator config should be overridable."""
    config = NornConfig(
        coordinator={"enabled": True, "activation_threshold": 1, "max_workers_per_phase": 3}
    )
    assert config.coordinator.enabled is True
    assert config.coordinator.activation_threshold == 1
    assert config.coordinator.max_workers_per_phase == 3
