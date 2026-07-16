"""Tests for vision-model resolution and image_read registry gating."""

from __future__ import annotations

from norn.cli.main import _build_flag_registry, _build_registry, _resolve_vision_model
from norn.core.config import NornConfig, RouterConfig, RouterTierConfig


def test_resolve_vision_model_legacy():
    cfg = NornConfig()
    cfg.llm.provider = "github_copilot"
    cfg.llm.model = "claude-sonnet-4.5"
    model, api_base = _resolve_vision_model(cfg)
    assert model == "github_copilot/claude-sonnet-4.5"


def test_resolve_vision_model_router_standard_tier():
    cfg = NornConfig()
    cfg.router = RouterConfig(
        enabled=True,
        tiers={
            "fast": RouterTierConfig(provider="ollama", model="small"),
            "standard": RouterTierConfig(provider="github_copilot", model="claude-sonnet-4.5"),
        },
    )
    model, _ = _resolve_vision_model(cfg)
    assert model == "github_copilot/claude-sonnet-4.5"


def test_image_read_registered_only_when_flag_on():
    cfg = NornConfig()
    cfg.flags.vision_tools = True
    fr = _build_flag_registry(cfg)
    reg = _build_registry(fr, vision_model="github_copilot/claude-sonnet-4.5")
    assert "image_read" in [t.name for t in reg.list_tools()]


def test_image_read_hidden_when_flag_off():
    cfg = NornConfig()
    cfg.flags.vision_tools = False
    fr = _build_flag_registry(cfg)
    reg = _build_registry(fr, vision_model="github_copilot/claude-sonnet-4.5")
    # Registered but gated off -> not exposed in the active tool list
    assert "image_read" not in [t.name for t in reg.list_tools()]


def test_image_read_absent_when_no_vision_model():
    cfg = NornConfig()
    cfg.flags.vision_tools = True
    fr = _build_flag_registry(cfg)
    reg = _build_registry(fr, vision_model=None)
    assert "image_read" not in [t.name for t in reg.list_tools()]
