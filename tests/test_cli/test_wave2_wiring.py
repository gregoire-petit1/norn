"""Guard: CLI construction wires wave-2 flags to the right callables.

Regression test for a mis-patched kwarg landing on _build_registry instead
of AgentLoop (TypeError at `norn run` with all wave-2 flags on).
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from norn.cli.main import _build_flag_registry, _build_registry, _build_task_notes
from norn.core.agent import AgentLoop
from norn.core.config import NornConfig


def _all_wave2_on() -> NornConfig:
    cfg = NornConfig()
    cfg.agent.task_notes = True
    cfg.agent.auto_plan = True
    cfg.agent.auto_verify = True
    cfg.agent.adaptive_rounds = True
    cfg.agent.stable_prompt = True
    cfg.flags.coderag = True
    cfg.flags.axi_output = True
    cfg.context.sliding_window = True
    return cfg


def test_build_registry_accepts_only_its_kwargs():
    cfg = _all_wave2_on()
    tn = _build_task_notes(cfg)
    # Must not raise TypeError on unexpected kwargs.
    reg = _build_registry(_build_flag_registry(cfg), config=cfg, task_notes=tn)
    names = {t.name for t in reg.list_tools()}
    assert "task_notes" in names
    assert {"coderag_search", "coderag_context", "coderag_explain"} <= names


def test_agentloop_accepts_full_cli_kwargs():
    """Mirror the exact kwargs the CLI passes to AgentLoop with everything on."""
    cfg = _all_wave2_on()
    tn = _build_task_notes(cfg)
    reg = _build_registry(_build_flag_registry(cfg), config=cfg, task_notes=tn)
    agent = AgentLoop(
        llm=AsyncMock(),
        registry=reg,
        cwd=".",
        max_tool_rounds=cfg.agent.max_tool_rounds,
        minify_tool_schemas=cfg.agent.minify_tool_schemas,
        stable_prompt=cfg.agent.stable_prompt,
        thread_invariants=cfg.agent.thread_invariants,
        max_tool_result_chars=cfg.agent.max_tool_result_chars,
        max_turn_output_chars=cfg.agent.max_turn_output_chars,
        env_bootstrap=False,
        repo_map=False,
        task_notes=tn,
        adaptive_rounds=cfg.agent.adaptive_rounds,
        max_round_extensions=cfg.agent.max_round_extensions,
        round_extension_factor=cfg.agent.round_extension_factor,
    )
    assert agent._adaptive_rounds is True
    assert agent._task_notes is tn
