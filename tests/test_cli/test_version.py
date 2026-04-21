"""Test that `norn version` skips logging bootstrap (Phase 9 C1)."""

from __future__ import annotations

from typer.testing import CliRunner

from norn.cli.main import app


def test_version_command_does_not_init_logging(monkeypatch) -> None:
    """norn version must be zero-cost: no config load, no logging init."""
    called = {"bootstrap": False, "config_load": False}

    def fake_bootstrap(*a, **kw):
        called["bootstrap"] = True

    def fake_load(*a, **kw):
        called["config_load"] = True
        raise RuntimeError("config should not be loaded for `version`")

    monkeypatch.setattr("norn.cli.main._bootstrap_logging", fake_bootstrap)
    monkeypatch.setattr("norn.cli.main.NornConfig.load", classmethod(fake_load))

    runner = CliRunner()
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0, f"version failed: {result.output}"
    assert "norn 0.1.0" in result.output
    assert called["bootstrap"] is False, "version must not call _bootstrap_logging"
    assert called["config_load"] is False, "version must not call NornConfig.load"
