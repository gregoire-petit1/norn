"""Tests for CodeRAG tool registration gating (SOTA v2, workstream B — W4.5)."""

from __future__ import annotations

from norn.cli.main import _build_flag_registry, _build_registry
from norn.core.config import NornConfig

_CODERAG_TOOLS = {"coderag_search", "coderag_context", "coderag_explain"}


def test_coderag_tools_registered_when_flag_on():
    cfg = NornConfig()
    cfg.flags.coderag = True
    fr = _build_flag_registry(cfg)
    reg = _build_registry(fr)
    assert _CODERAG_TOOLS <= {t.name for t in reg.list_tools()}


def test_coderag_tools_hidden_when_flag_off():
    cfg = NornConfig()
    cfg.flags.coderag = False
    fr = _build_flag_registry(cfg)
    reg = _build_registry(fr)
    assert not (_CODERAG_TOOLS & {t.name for t in reg.list_tools()})


def test_coderag_flag_follows_config():
    cfg = NornConfig()
    cfg.flags.coderag = True
    assert _build_flag_registry(cfg).is_enabled("coderag")
    cfg.flags.coderag = False
    assert not _build_flag_registry(cfg).is_enabled("coderag")
