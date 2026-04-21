"""Tests for LoggingConfig."""

import pydantic
import pytest

from norn.core.config import LoggingConfig, NornConfig


def test_logging_config_defaults():
    cfg = LoggingConfig()
    assert cfg.enabled is True
    assert cfg.level == "INFO"
    assert cfg.output == "both"
    assert cfg.file_dir == "~/.norn/logs"
    assert cfg.include_cost is True
    assert "api_key" in cfg.redact_keys


def test_norn_config_includes_logging():
    cfg = NornConfig()
    assert cfg.logging.enabled is True
    assert cfg.logging.level == "INFO"


def test_logging_config_rejects_invalid_level():
    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(level="TRACE")


def test_logging_config_rejects_invalid_output():
    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(output="syslog")


def test_norn_config_from_yaml_with_logging(tmp_path):
    """Partial logging block in YAML preserves unspecified defaults."""
    p = tmp_path / "c.yaml"
    p.write_text("logging:\n  level: DEBUG\n  output: console\n")
    cfg = NornConfig.from_yaml(p)
    assert cfg.logging.level == "DEBUG"
    assert cfg.logging.output == "console"
    assert cfg.logging.enabled is True  # default preserved
    assert "api_key" in cfg.logging.redact_keys  # default preserved
