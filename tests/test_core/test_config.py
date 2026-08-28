"""Tests for configuration system."""

from norn.core.config import (
    NornConfig,
    PermissionMode,
    RouterConfig,
    RouterTierConfig,
)


def test_default_config():
    config = NornConfig()
    assert config.llm.provider == "ollama"
    assert config.permissions.mode == PermissionMode.INTERACTIVE


def test_bench_config_defaults():
    from norn.core.config import BenchConfig

    cfg = BenchConfig()
    assert cfg.tasks_dir == "benchmarks/tasks"
    assert cfg.results_dir == "benchmarks/results"
    assert cfg.reports_dir == "benchmarks/reports"
    assert cfg.default_n_runs == 1
    assert cfg.default_timeout_seconds == 300
    assert cfg.judge.enabled is True
    assert cfg.judge.pass_threshold == 7.0
    assert cfg.judge.model == "anthropic/claude-sonnet-4"


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


def test_router_config_defaults():
    config = NornConfig()
    assert config.router.enabled is False
    assert config.router.tiers == {}


def test_router_tier_config():
    tier = RouterTierConfig(provider="ollama", model="qwen2.5-coder:7b")
    assert tier.provider == "ollama"
    assert tier.model == "qwen2.5-coder:7b"
    assert tier.api_base is None


def test_router_config_direct_construction():
    """RouterConfig can be built directly without going through NornConfig."""
    cfg = RouterConfig(
        enabled=True,
        tiers={"fast": RouterTierConfig(provider="ollama", model="qwen2.5-coder:7b")},
    )
    assert cfg.enabled is True
    assert cfg.tiers["fast"].provider == "ollama"


def test_router_config_with_tiers():
    config = NornConfig(
        router={
            "enabled": True,
            "tiers": {
                "fast": {"provider": "ollama", "model": "qwen2.5-coder:7b"},
                "standard": {"provider": "openrouter", "model": "stepfun/step-3.5-flash:free"},
                "powerful": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            },
        }
    )
    assert config.router.enabled is True
    assert config.router.tiers["fast"].model == "qwen2.5-coder:7b"
    assert config.router.tiers["standard"].provider == "openrouter"
    assert config.router.tiers["powerful"].model == "anthropic/claude-sonnet-4"


def test_router_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "router:\n"
        "  enabled: true\n"
        "  tiers:\n"
        "    fast:\n"
        "      provider: ollama\n"
        "      model: qwen2.5-coder:7b\n"
    )
    config = NornConfig.from_yaml(config_file)
    assert config.router.enabled is True
    assert config.router.tiers["fast"].provider == "ollama"


# --------------------------------------------------------------------------- #
# G.3 — prompt cache config knob
# --------------------------------------------------------------------------- #


def test_default_llm_prompt_cache_is_true():
    """prompt_cache defaults to True for opt-out-by-config behaviour."""
    config = NornConfig()
    assert config.llm.prompt_cache is True


def test_llm_prompt_cache_can_be_disabled_via_yaml(tmp_path):
    """YAML can disable prompt cache explicitly."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  provider: anthropic\n  model: anthropic/claude-sonnet-4\n  prompt_cache: false\n"
    )
    config = NornConfig.from_yaml(config_file)
    assert config.llm.prompt_cache is False


# --------------------------------------------------------------------------- #
# NORN_CONFIG_OVERRIDES env var
# --------------------------------------------------------------------------- #


def test_config_overrides_from_env(monkeypatch):
    """NORN_CONFIG_OVERRIDES JSON merges nested fields into config."""
    import json

    overrides = {
        "logging": {"file_dir": "/tmp/test", "output": "file"},
        "permissions": {"mode": "strict"},
    }
    monkeypatch.setenv("NORN_CONFIG_OVERRIDES", json.dumps(overrides))
    config = NornConfig()
    config.apply_env_overrides()
    assert config.logging.file_dir == "/tmp/test"
    assert config.logging.output == "file"
    assert config.permissions.mode == PermissionMode.STRICT


def test_config_overrides_invalid_json_ignored(monkeypatch):
    """Malformed JSON in NORN_CONFIG_OVERRIDES must not raise."""
    monkeypatch.setenv("NORN_CONFIG_OVERRIDES", "not-json{{{")
    config = NornConfig()
    config.apply_env_overrides()
    # Defaults unchanged
    assert config.permissions.mode == PermissionMode.INTERACTIVE
