"""Tests for RouterProvider."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from norn.core.config import RouterConfig, RouterTierConfig
from norn.core.models import LLMResponse, Message, Role, TokenUsage
from norn.core.router import RouterProvider, Tier, _classify_complexity


def _msg(content: str) -> Message:
    return Message(role=Role.USER, content=content)


def _history(n: int) -> list[Message]:
    return [_msg("msg") for _ in range(n)]


# ── Tier classification ────────────────────────────────────────────────────


def test_classify_simple_prompt_is_fast():
    messages = [_msg("hello")]
    assert _classify_complexity(messages, []) == Tier.FAST


def test_classify_long_prompt_is_standard():
    messages = [_msg("x" * 501)]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_long_history_is_standard():
    messages = _history(11)
    assert _classify_complexity(messages, []) == Tier.STANDARD


@pytest.mark.parametrize(
    "keyword",
    ["architect", "design", "refactor", "optimize", "analyze", "compare"],
)
def test_classify_keyword_in_prompt_is_standard(keyword: str):
    messages = [_msg(f"please {keyword} this")]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_keyword_case_insensitive():
    messages = [_msg("REFACTOR this module")]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_keyword_with_trailing_punctuation():
    """Regression: 'refactor.' (with punctuation) must still match the keyword."""
    messages = [_msg("please refactor.")]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_keyword_substring_does_not_match():
    """'refactoring' (substring) must NOT match 'refactor' (word boundary)."""
    messages = [_msg("refactoring code")]
    assert _classify_complexity(messages, []) == Tier.FAST


def test_classify_many_tools_is_standard():
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    messages = [_msg("hello")]
    assert _classify_complexity(messages, tools) == Tier.STANDARD


def test_classify_multiple_signals_is_powerful():
    # long prompt + keyword + long history = score 3 → POWERFUL
    messages = _history(11) + [_msg("x" * 501 + " please refactor")]
    assert _classify_complexity(messages, []) == Tier.POWERFUL


def test_classify_empty_messages_is_fast():
    assert _classify_complexity([], []) == Tier.FAST


def test_classify_score_boundary_standard():
    # score exactly 2: long history + many tools → STANDARD
    messages = _history(11)
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    assert _classify_complexity(messages, tools) == Tier.STANDARD


def test_classify_score_boundary_powerful():
    # score 3: long history + many tools + keyword → POWERFUL
    messages = _history(11) + [_msg("please analyze")]
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    assert _classify_complexity(messages, tools) == Tier.POWERFUL


# ── RouterProvider ─────────────────────────────────────────────────────────


def _make_router_config() -> RouterConfig:
    return RouterConfig(
        enabled=True,
        tiers={
            "fast": RouterTierConfig(provider="ollama", model="qwen2.5-coder:7b"),
            "standard": RouterTierConfig(
                provider="openrouter", model="stepfun/step-3.5-flash:free"
            ),
            "powerful": RouterTierConfig(
                provider="openrouter", model="anthropic/claude-sonnet-4"
            ),
        },
    )


def _make_response(content: str = "ok") -> LLMResponse:
    return LLMResponse(content=content, usage=TokenUsage())


@pytest.mark.asyncio
async def test_router_complete_uses_fast_tier_for_simple_prompt():
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.FAST],
        "complete",
        new_callable=AsyncMock,
        return_value=_make_response("fast response"),
    ) as mock_fast:
        result = await router.complete([_msg("hello")])
        assert result.content == "fast response"
        mock_fast.assert_called_once()


@pytest.mark.asyncio
async def test_router_complete_uses_powerful_tier_for_complex_prompt():
    messages = _history(11) + [_msg("x" * 501 + " please refactor and analyze")]
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.POWERFUL],
        "complete",
        new_callable=AsyncMock,
        return_value=_make_response("powerful response"),
    ) as mock_powerful:
        result = await router.complete(messages)
        assert result.content == "powerful response"
        mock_powerful.assert_called_once()


@pytest.mark.asyncio
async def test_router_fallback_on_http_error():
    config = _make_router_config()
    router = RouterProvider(config)

    with (
        patch.object(
            router._providers[Tier.FAST],
            "complete",
            new_callable=AsyncMock,
            side_effect=Exception("402 Payment Required"),
        ),
        patch.object(
            router._providers[Tier.STANDARD],
            "complete",
            new_callable=AsyncMock,
            return_value=_make_response("standard response"),
        ) as mock_standard,
    ):
        result = await router.complete([_msg("hello")])
        assert result.content == "standard response"
        mock_standard.assert_called_once()


@pytest.mark.asyncio
async def test_router_fallback_on_timeout():
    config = _make_router_config()
    router = RouterProvider(config)

    with (
        patch.object(
            router._providers[Tier.FAST],
            "complete",
            new_callable=AsyncMock,
            side_effect=Exception("connection timeout"),
        ),
        patch.object(
            router._providers[Tier.STANDARD],
            "complete",
            new_callable=AsyncMock,
            return_value=_make_response("ok"),
        ) as mock_standard,
    ):
        result = await router.complete([_msg("hello")])
        assert result.content == "ok"
        mock_standard.assert_called_once()


@pytest.mark.asyncio
async def test_router_no_fallback_on_non_technical_error():
    """Non-technical errors (e.g. validation) must NOT trigger fallback."""
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.FAST],
        "complete",
        new_callable=AsyncMock,
        side_effect=ValueError("invalid argument"),
    ), pytest.raises(ValueError, match="invalid argument"):
        await router.complete([_msg("hello")])


@pytest.mark.asyncio
async def test_router_raises_when_all_tiers_exhausted():
    """If all tiers fail with technical errors, re-raise the last error."""
    config = _make_router_config()
    router = RouterProvider(config)

    with (
        patch.object(
            router._providers[Tier.FAST],
            "complete",
            new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway"),
        ),
        patch.object(
            router._providers[Tier.STANDARD],
            "complete",
            new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway"),
        ),
        patch.object(
            router._providers[Tier.POWERFUL],
            "complete",
            new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway"),
        ),pytest.raises(Exception, match="502")
    ):
        await router.complete([_msg("hello")])


@pytest.mark.asyncio
async def test_router_complete_with_tier_override():
    """When tier_override is given, skip classification and use that tier."""
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.POWERFUL],
        "complete",
        new_callable=AsyncMock,
        return_value=_make_response("powerful override"),
    ) as mock_powerful:
        result = await router.complete([_msg("hello")], tier_override=Tier.POWERFUL)
        assert result.content == "powerful override"
        mock_powerful.assert_called_once()


def test_router_provider_default_tier_attribute():
    """RouterProvider stores default_tier and uses it when no override given."""
    config = _make_router_config()
    router = RouterProvider(config, default_tier=Tier.POWERFUL)
    assert router.default_tier == Tier.POWERFUL


@pytest.mark.asyncio
async def test_router_default_tier_used_when_no_override():
    config = _make_router_config()
    router = RouterProvider(config, default_tier=Tier.POWERFUL)

    with patch.object(
        router._providers[Tier.POWERFUL],
        "complete",
        new_callable=AsyncMock,
        return_value=_make_response("from default"),
    ) as mock_powerful:
        # Simple prompt would normally route to FAST, but default_tier=POWERFUL forces POWERFUL
        result = await router.complete([_msg("hello")])
        assert result.content == "from default"
        mock_powerful.assert_called_once()


@pytest.mark.asyncio
async def test_router_fallback_skips_missing_tiers():
    """If only standard+powerful are configured, FAST classification jumps to STANDARD."""
    config = RouterConfig(
        enabled=True,
        tiers={
            "standard": RouterTierConfig(provider="openrouter", model="m1"),
            "powerful": RouterTierConfig(provider="openrouter", model="m2"),
        },
    )
    router = RouterProvider(config)
    # Simple prompt would classify as FAST, but FAST is missing → should use STANDARD
    with patch.object(
        router._providers[Tier.STANDARD],
        "complete",
        new_callable=AsyncMock,
        return_value=_make_response("from standard"),
    ) as mock_standard:
        result = await router.complete([_msg("hello")])
        assert result.content == "from standard"
        mock_standard.assert_called_once()
