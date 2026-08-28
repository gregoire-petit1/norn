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


# --------------------------------------------------------------------------- #
# SOTA v2 workstream D — CACHE_BREAK split + streaming cache metrics
# --------------------------------------------------------------------------- #

from norn.core.prompts import CACHE_BREAK  # noqa: E402


def test_apply_cache_markers_splits_on_cache_break():
    static = "instructions + env + repo map"
    volatile = "## Persistent Memory\n\nremember X"
    msgs = [{"role": "system", "content": static + CACHE_BREAK + volatile}]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=True)
    blocks = out_msgs[0]["content"]
    assert isinstance(blocks, list)
    assert len(blocks) == 2
    assert blocks[0]["text"] == static
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    assert blocks[1]["text"] == volatile
    assert "cache_control" not in blocks[1]
    # Sentinel never reaches the provider
    assert CACHE_BREAK not in blocks[0]["text"] + blocks[1]["text"]


def test_apply_cache_markers_disabled_strips_cache_break():
    msgs = [{"role": "system", "content": "static" + CACHE_BREAK + "volatile"}]
    out_msgs, _ = _apply_cache_markers(msgs, None, enabled=False)
    content = out_msgs[0]["content"]
    assert isinstance(content, str)
    assert CACHE_BREAK not in content
    assert "static" in content and "volatile" in content


@pytest.mark.asyncio
async def test_litellm_provider_strips_cache_break_on_non_caching_model():
    """Non-eligible model → sentinel stripped, plain string content."""
    fake = AsyncMock(return_value=_minimal_fake_response())
    provider = LiteLLMProvider(
        model="ollama/qwen2.5-coder:14b",
        prompt_cache=True,
        completion_fn=fake,
    )
    await provider.complete(
        messages=[
            Message(role=Role.SYSTEM, content="static" + CACHE_BREAK + "volatile"),
            Message(role=Role.USER, content="hi"),
        ],
        tools=None,
    )
    sys_content = fake.await_args.kwargs["messages"][0]["content"]
    assert isinstance(sys_content, str)
    assert CACHE_BREAK not in sys_content


def _fake_stream_chunks(cache_read: int = 0):
    """Two chunks: one delta, one final with usage."""
    c1 = MagicMock()
    d1 = MagicMock()
    d1.content = "hel"
    d1.tool_calls = None
    ch1 = MagicMock()
    ch1.delta = d1
    ch1.finish_reason = None
    c1.choices = [ch1]
    c1.usage = None

    c2 = MagicMock()
    d2 = MagicMock()
    d2.content = "lo"
    d2.tool_calls = None
    ch2 = MagicMock()
    ch2.delta = d2
    ch2.finish_reason = "stop"
    c2.choices = [ch2]
    c2.usage = MagicMock(
        prompt_tokens=100,
        completion_tokens=5,
        total_tokens=105,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=0,
    )
    return [c1, c2]


async def _aiter(items):
    for item in items:
        yield item


@pytest.mark.asyncio
async def test_stream_emits_llm_complete_event_with_cache_metrics(tmp_path):
    """stream() logs an llm.complete event carrying token + cache metrics."""
    import json
    import logging as _logging

    from norn.core.config import LoggingConfig
    from norn.observability.logger import init_logging, new_session

    init_logging(LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path)))
    new_session()
    try:
        fake = AsyncMock(return_value=_aiter(_fake_stream_chunks(cache_read=90)))
        provider = LiteLLMProvider(
            model="anthropic/claude-sonnet-4",
            prompt_cache=False,
            completion_fn=fake,
        )
        chunks = [
            c
            async for c in provider.stream(
                messages=[Message(role=Role.USER, content="hi")],
            )
        ]
        assert chunks[-1].done
        assert chunks[-1].usage is not None
        assert chunks[-1].usage.cache_read_tokens == 90

        from datetime import UTC, datetime

        log_file = tmp_path / f"{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        events = [json.loads(line) for line in log_file.read_text().splitlines() if line.strip()]
        llm_events = [e for e in events if e.get("event") == "llm.complete"]
        assert len(llm_events) == 1
        assert llm_events[0]["stream"] is True
        assert llm_events[0]["cache_read_tokens"] == 90
        assert llm_events[0]["prompt_tokens"] == 100
        assert llm_events[0]["success"] is True
    finally:
        root = _logging.getLogger()
        for h in list(root.handlers):
            h.close()
            root.removeHandler(h)


@pytest.mark.asyncio
async def test_stream_options_gated_by_provider():
    """OpenAI-compatible providers get stream_options; anthropic does not."""
    fake = AsyncMock(return_value=_aiter(_fake_stream_chunks()))
    provider = LiteLLMProvider(model="groq/llama-3.3-70b", prompt_cache=False, completion_fn=fake)
    async for _ in provider.stream(messages=[Message(role=Role.USER, content="hi")]):
        pass
    assert fake.await_args.kwargs["stream_options"] == {"include_usage": True}

    fake2 = AsyncMock(return_value=_aiter(_fake_stream_chunks()))
    provider2 = LiteLLMProvider(
        model="anthropic/claude-sonnet-4", prompt_cache=False, completion_fn=fake2
    )
    async for _ in provider2.stream(messages=[Message(role=Role.USER, content="hi")]):
        pass
    assert "stream_options" not in fake2.await_args.kwargs
