"""Tests for local token accounting fallback (wave 3 — #1 instrumentation)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from norn.core.llm import LiteLLMProvider, _local_token_usage
from norn.core.models import Message, Role, ToolCall

_MSGS = [{"role": "user", "content": "fix the bug"}]


def test_local_usage_counts_prompt_and_completion():
    u = _local_token_usage("github_copilot/claude-sonnet-4.5", _MSGS, None, "done", None)
    assert u is not None
    assert u.prompt_tokens > 0
    assert u.completion_tokens > 0
    assert u.total_tokens == u.prompt_tokens + u.completion_tokens


def test_local_usage_tools_add_prompt_tokens():
    tools = [{"type": "function", "function": {"name": "bash", "description": "x" * 200}}]
    bare = _local_token_usage("anthropic/claude-sonnet-4", _MSGS, None, None, None)
    with_tools = _local_token_usage("anthropic/claude-sonnet-4", _MSGS, tools, None, None)
    assert with_tools.prompt_tokens > bare.prompt_tokens


def test_local_usage_counts_tool_calls_as_output():
    calls = [ToolCall(id="1", name="bash", arguments={"command": "pytest -q"})]
    u = _local_token_usage("anthropic/claude-sonnet-4", _MSGS, None, None, calls)
    assert u.completion_tokens > 0


def test_local_usage_returns_none_when_tokenizer_fails(monkeypatch):
    import norn.core.llm as llm_mod

    monkeypatch.setattr(
        llm_mod.litellm, "token_counter", MagicMock(side_effect=RuntimeError("no tokenizer"))
    )
    assert _local_token_usage("x/y", _MSGS, None, "z", None) is None


def _resp(with_usage: bool):
    r = MagicMock()
    ch = MagicMock()
    ch.message.content = "answer"
    ch.message.tool_calls = None
    ch.finish_reason = "stop"
    r.choices = [ch]
    r.usage = (
        MagicMock(
            prompt_tokens=11,
            completion_tokens=3,
            total_tokens=14,
            cache_read_input_tokens=0,
            cache_creation_input_tokens=0,
        )
        if with_usage
        else None
    )
    return r


@pytest.mark.asyncio
async def test_provider_usage_wins_and_is_tagged(tmp_path):
    fake = AsyncMock(return_value=_resp(True))
    p = LiteLLMProvider(model="anthropic/claude-sonnet-4", completion_fn=fake, prompt_cache=False)
    out = await p.complete(messages=[Message(role=Role.USER, content="hi")])
    assert out.usage.prompt_tokens == 11  # provider value, not an estimate


@pytest.mark.asyncio
async def test_local_fallback_fills_zeros():
    """No provider usage (Copilot case) → local estimate rather than zeros."""
    fake = AsyncMock(return_value=_resp(False))
    p = LiteLLMProvider(
        model="github_copilot/claude-sonnet-4.5", completion_fn=fake, prompt_cache=False
    )
    out = await p.complete(messages=[Message(role=Role.USER, content="hi")])
    assert out.usage.prompt_tokens > 0
    assert out.usage.total_tokens > 0


@pytest.mark.asyncio
async def test_local_fallback_can_be_disabled():
    fake = AsyncMock(return_value=_resp(False))
    p = LiteLLMProvider(
        model="github_copilot/claude-sonnet-4.5",
        completion_fn=fake,
        prompt_cache=False,
        count_tokens_locally=False,
    )
    out = await p.complete(messages=[Message(role=Role.USER, content="hi")])
    assert out.usage.total_tokens == 0


@pytest.mark.asyncio
async def test_token_source_logged(tmp_path):
    """llm.complete carries token_source so estimates are never mistaken for measurements."""
    import json
    import logging as _logging
    from datetime import UTC, datetime

    from norn.core.config import LoggingConfig
    from norn.observability.logger import init_logging, new_session

    init_logging(LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path)))
    new_session()
    try:
        fake = AsyncMock(return_value=_resp(False))
        p = LiteLLMProvider(
            model="github_copilot/claude-sonnet-4.5", completion_fn=fake, prompt_cache=False
        )
        await p.complete(messages=[Message(role=Role.USER, content="hi")])
        log = tmp_path / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        events = [json.loads(x) for x in log.read_text().splitlines() if x.strip()]
        ev = [e for e in events if e.get("event") == "llm.complete"][0]
        assert ev["token_source"] == "local"
        assert ev["prompt_tokens"] > 0
    finally:
        root = _logging.getLogger()
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)


def test_drop_params_enabled_by_default():
    """Reasoning models (gpt-5*) reject temperature=0.0; litellm must drop it.

    Regression: 13/13 tb2 trials died instantly with UnsupportedParamsError
    because Norn sends temperature=0.0 to every model.
    """
    import norn.core.llm as llm_mod

    llm_mod.litellm.drop_params = False
    LiteLLMProvider(model="github_copilot/gpt-5.3-codex")
    assert llm_mod.litellm.drop_params is True


def test_drop_params_can_be_disabled():
    import norn.core.llm as llm_mod

    llm_mod.litellm.drop_params = False
    LiteLLMProvider(model="x/y", drop_unsupported_params=False)
    assert llm_mod.litellm.drop_params is False
    llm_mod.litellm.drop_params = True  # restore for other tests
