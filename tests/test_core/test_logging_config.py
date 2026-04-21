"""Tests for LoggingConfig."""

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
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(level="TRACE")


def test_logging_config_rejects_invalid_output():
    import pydantic
    import pytest

    with pytest.raises(pydantic.ValidationError):
        LoggingConfig(output="syslog")
