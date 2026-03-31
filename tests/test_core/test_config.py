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
