"""Tests for prompt-cache helpers and integration in LiteLLMProvider.

Phase 9 v2 — Workstream G.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from norn.core.llm import LiteLLMProvider, _apply_cache_markers, _supports_prompt_cache
from norn.core.models import Message, Role

# --------------------------------------------------------------------------- #
# G.1 — _supports_prompt_cache eligibility
# --------------------------------------------------------------------------- #


def test_supports_prompt_cache_anthropic_direct():
    assert _supports_prompt_cache("anthropic/claude-sonnet-4")


def test_supports_prompt_cache_anthropic_via_openrouter():
    assert _supports_prompt_cache("openrouter/anthropic/claude-sonnet-4")


def test_supports_prompt_cache_openai_4o():
    assert _supports_prompt_cache("openai/gpt-4o")


def test_supports_prompt_cache_openai_41():
    assert _supports_prompt_cache("openai/gpt-4.1")


def test_supports_prompt_cache_unsupported_ollama():
    assert not _supports_prompt_cache("ollama/qwen2.5-coder:14b")


def test_supports_prompt_cache_unsupported_openrouter_other():
    assert not _supports_prompt_cache("openrouter/stepfun/step-3.5-flash:free")


# --------------------------------------------------------------------------- #
# G.1 — _apply_cache_markers behaviour
# --------------------------------------------------------------------------- #


def test_apply_cache_markers_disabled_is_noop():
    msgs = [{"role": "system", "content": "you are an agent"}]
    tools = [{"type": "function", "function": {"name": "bash"}}]
    out_msgs, out_tools = _apply_cache_markers(msgs, tools, enabled=False)
    assert out_msgs == msgs
    assert out_tools == tools


def test_apply_cache_markers_tags_system_prompt():
    msgs = [
        {"role": "system", "content": "you are an agent"},
        {"role": "user", "content": "hi"},
    ]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert isinstance(out_msgs[0]["content"], list)
    assert out_msgs[0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert out_msgs[0]["content"][0]["text"] == "you are an agent"
    # User message untouched
    assert out_msgs[1] == msgs[1]


def test_apply_cache_markers_tags_last_tool():
    tools = [
        {"type": "function", "function": {"name": "bash"}},
        {"type": "function", "function": {"name": "file_read"}},
    ]
    _, out_tools = _apply_cache_markers([], tools, enabled=True)
    assert out_tools[0] == tools[0]  # untouched
    assert out_tools[1]["cache_control"] == {"type": "ephemeral"}


def test_apply_cache_markers_no_system_no_crash():
    msgs = [{"role": "user", "content": "hi"}]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    assert out_msgs == msgs


def test_apply_cache_markers_empty_messages():
    out_msgs, out_tools = _apply_cache_markers([], None, enabled=True)
    assert out_msgs == []
    assert out_tools is None


# --------------------------------------------------------------------------- #
# G.2 — LiteLLMProvider integration
# --------------------------------------------------------------------------- #


def _minimal_fake_response(
    cache_read: int = 0,
    cache_creation: int = 0,
) -> MagicMock:
    """Build a minimal fake litellm completion response."""
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = "ok"
    choice.message.tool_calls = None
    choice.finish_reason = "stop"
    resp.choices = [choice]
    resp.usage = MagicMock(
        prompt_tokens=1000,
        completion_tokens=10,
        total_tokens=1010,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_creation,
    )
    return resp


@pytest.mark.asyncio
async def test_litellm_provider_passes_cache_markers_to_litellm():
    """Eligible model + prompt_cache=True → litellm receives cache_control markers."""
    fake = AsyncMock(return_value=_minimal_fake_response())
    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-4",
        prompt_cache=True,
        completion_fn=fake,
    )
    await provider.complete(
        messages=[
            Message(role=Role.SYSTEM, content="hello"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=None,
    )
    call_kwargs = fake.await_args.kwargs
    sys_content = call_kwargs["messages"][0]["content"]
    assert isinstance(sys_content, list)
    assert sys_content[0]["cache_control"] == {"type": "ephemeral"}


@pytest.mark.asyncio
async def test_litellm_provider_skips_markers_on_unsupported_model():
    """Non-eligible model → no cache_control markers, content stays a plain string."""
    fake = AsyncMock(return_value=_minimal_fake_response())
    provider = LiteLLMProvider(
        model="ollama/qwen2.5-coder:14b",
        prompt_cache=True,
        completion_fn=fake,
    )
    await provider.complete(
        messages=[
            Message(role=Role.SYSTEM, content="hello"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=None,
    )
    call_kwargs = fake.await_args.kwargs
    sys_content = call_kwargs["messages"][0]["content"]
    assert sys_content == "hello"


@pytest.mark.asyncio
async def test_litellm_provider_skips_markers_when_disabled():
    """prompt_cache=False → no markers even on an eligible model."""
    fake = AsyncMock(return_value=_minimal_fake_response())
    provider = LiteLLMProvider(
        model="anthropic/claude-sonnet-4",
        prompt_cache=False,
        completion_fn=fake,
    )
    await provider.complete(
        messages=[
            Message(role=Role.SYSTEM, content="hello"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=None,
    )
    call_kwargs = fake.await_args.kwargs
    sys_content = call_kwargs["messages"][0]["content"]
    assert sys_content == "hello"


@pytest.mark.asyncio
async def test_litellm_provider_emits_cache_observability_fields(tmp_path):
    """When usage carries cache stats, llm.complete event includes them."""
    import json
    import logging as _logging

    from norn.core.config import LoggingConfig
    from norn.observability.logger import init_logging, new_session

    init_logging(LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path)))
    new_session()
    try:
        fake = AsyncMock(return_value=_minimal_fake_response(cache_read=800, cache_creation=200))
        provider = LiteLLMProvider(
            model="anthropic/claude-sonnet-4",
            prompt_cache=True,
            completion_fn=fake,
        )
        await provider.complete(
            messages=[
                Message(role=Role.SYSTEM, content="hello"),
                Message(role=Role.USER, content="hi"),
            ],
            tools=None,
        )

        # Find today's UTC log file (sink rotates by UTC date)
        from datetime import UTC, datetime

        log_file = tmp_path / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        events = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        llm_events = [e for e in events if e.get("event") == "llm.complete"]
        assert len(llm_events) == 1
        assert llm_events[0]["cache_read_tokens"] == 800
        assert llm_events[0]["cache_creation_tokens"] == 200
    finally:
        root = _logging.getLogger()
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)
