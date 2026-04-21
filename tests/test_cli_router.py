"""Tests for --model CLI option and _build_provider routing logic."""

from __future__ import annotations

import pytest
import typer
from typer.testing import CliRunner

from norn.cli.main import _build_provider, app
from norn.core.config import NornConfig, RouterConfig, RouterTierConfig
from norn.core.llm import LiteLLMProvider
from norn.core.router import RouterProvider, Tier

runner = CliRunner()


# ── _build_provider ────────────────────────────────────────────────────────


def test_build_provider_returns_litellm_when_router_disabled():
    """Default config has router.enabled=False → returns LiteLLMProvider."""
    config = NornConfig()
    provider = _build_provider(config)
    assert isinstance(provider, LiteLLMProvider)


def test_build_provider_returns_router_when_router_enabled():
    config = NornConfig(
        router=RouterConfig(
            enabled=True,
            tiers={"fast": RouterTierConfig(provider="ollama", model="qwen:7b")},
        )
    )
    provider = _build_provider(config)
    assert isinstance(provider, RouterProvider)
    assert provider.default_tier is None


def test_build_provider_router_with_valid_model_override():
    config = NornConfig(
        router=RouterConfig(
            enabled=True,
            tiers={
                "fast": RouterTierConfig(provider="ollama", model="qwen:7b"),
                "powerful": RouterTierConfig(provider="openrouter", model="claude-sonnet-4"),
            },
        )
    )
    provider = _build_provider(config, model_override="powerful")
    assert isinstance(provider, RouterProvider)
    assert provider.default_tier == Tier.POWERFUL


def test_build_provider_router_ignores_invalid_model_override():
    config = NornConfig(
        router=RouterConfig(
            enabled=True,
            tiers={"fast": RouterTierConfig(provider="ollama", model="qwen:7b")},
        )
    )
    provider = _build_provider(config, model_override="lightning")
    assert isinstance(provider, RouterProvider)
    # Invalid tier → default_tier stays None (warning printed)
    assert provider.default_tier is None


def test_build_provider_legacy_ignores_model_override():
    """When router is disabled, model_override is silently ignored (legacy path)."""
    config = NornConfig()
    provider = _build_provider(config, model_override="fast")
    assert isinstance(provider, LiteLLMProvider)


# ── CLI --model option ─────────────────────────────────────────────────────


def test_chat_help_includes_model_option():
    result = runner.invoke(app, ["chat", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_run_help_includes_model_option():
    result = runner.invoke(app, ["run", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_dream_help_includes_model_option():
    result = runner.invoke(app, ["dream", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


def test_coordinate_help_includes_model_option():
    result = runner.invoke(app, ["coordinate", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output


# ── End-to-end plumbing: --model actually reaches _build_provider ──────────


@pytest.mark.parametrize("command", ["chat", "run", "dream", "coordinate"])
def test_command_passes_model_override_to_build_provider(monkeypatch, command):
    """--model fast must flow through to _build_provider on every command."""
    captured: dict[str, str | None] = {}

    def fake_build(config, model_override=None):
        captured["model"] = model_override
        raise typer.Exit(0)  # short-circuit before the rest of the command runs

    monkeypatch.setattr("norn.cli.main._build_provider", fake_build)

    # `coordinate` gates on config.coordinator.enabled BEFORE calling _build_provider.
    # Force a config that enables it so the test actually exercises the plumbing path.
    from norn.core.config import CoordinatorConfig, NornConfig

    def fake_load() -> NornConfig:
        return NornConfig(coordinator=CoordinatorConfig(enabled=True))

    monkeypatch.setattr("norn.cli.main.NornConfig.load", staticmethod(fake_load))

    args = [command, "--model", "fast"]
    if command in ("run", "coordinate"):
        args.append("ping")  # required positional

    runner.invoke(app, args)
    assert captured.get("model") == "fast"


# ── Warning is printed for invalid --model ─────────────────────────────────


def test_build_provider_invalid_model_prints_warning(capsys):
    config = NornConfig(
        router=RouterConfig(
            enabled=True,
            tiers={"fast": RouterTierConfig(provider="ollama", model="qwen:7b")},
        )
    )
    _build_provider(config, model_override="lightning")
    captured = capsys.readouterr()
    assert "lightning" in captured.out or "lightning" in captured.err
    assert "Unknown tier" in captured.out or "Unknown tier" in captured.err


# ── Case-insensitivity ─────────────────────────────────────────────────────


@pytest.mark.parametrize("input_str", ["FAST", "Fast", "fAsT"])
def test_build_provider_model_override_is_case_insensitive(input_str):
    config = NornConfig(
        router=RouterConfig(
            enabled=True,
            tiers={"fast": RouterTierConfig(provider="ollama", model="qwen:7b")},
        )
    )
    provider = _build_provider(config, model_override=input_str)
    assert isinstance(provider, RouterProvider)
    assert provider.default_tier == Tier.FAST
