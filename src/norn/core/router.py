"""Tier classification primitives for the RouterProvider (Phase 7)."""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
from enum import StrEnum
from typing import TYPE_CHECKING

import structlog

from norn.core.llm import LiteLLMProvider
from norn.observability import EventName, get_logger

_logger = logging.getLogger(__name__)
_log = get_logger(__name__)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from norn.core.config import RouterConfig
    from norn.core.models import LLMResponse, Message, StreamChunk


# Tier-classification thresholds (tunable heuristics).
_LONG_PROMPT_CHARS = 500
_LONG_HISTORY_TURNS = 10
_MANY_TOOLS = 8
_STANDARD_MAX_SCORE = 2  # score <= this → STANDARD; above → POWERFUL

_KEYWORD_RE = re.compile(
    r"\b(?:architect|design|refactor|optimize|analyze|compare)\b",
    re.IGNORECASE,
)

# Domain patterns that benefit from POWERFUL tier (Phase 10)
_DOMAIN_BOOST_PATTERNS: dict[str, re.Pattern[str]] = {
    "architecture": re.compile(
        r"\b(?:system design|microservice|database schema|API design|distributed)\b", re.IGNORECASE
    ),
    "security": re.compile(
        r"\b(?:vulnerabilit|injection|XSS|CSRF|auth bypass|CVE|exploit)\b", re.IGNORECASE
    ),
    "ml_training": re.compile(
        r"\b(?:training loop|loss function|gradient|backprop|hyperparameter)\b", re.IGNORECASE
    ),
    "debugging": re.compile(
        r"\b(?:stack trace|segfault|deadlock|race condition|memory leak)\b", re.IGNORECASE
    ),
}

_SIMPLE_PATTERN = re.compile(r"^(?:what|how|where|when|why|which)\s.{5,80}\?$", re.IGNORECASE)


class Tier(StrEnum):
    """Routing tier — FAST (cheap/quick), STANDARD (default), POWERFUL (complex tasks)."""

    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


def _score_complexity(
    messages: list[Message],
    tools: list[dict],
    *,
    domain_routing: bool = True,
) -> tuple[int, dict[str, bool]]:
    """Compute the complexity score and the individual signals that contributed.

    Returns ``(score, signals)`` where ``signals`` is a flat dict of booleans
    suitable for inclusion in a ``routing.decision`` event.
    """
    last_content = messages[-1].content if messages else None
    long_prompt = bool(last_content and len(last_content) > _LONG_PROMPT_CHARS)
    complex_keyword = bool(last_content and _KEYWORD_RE.search(last_content))
    long_history = len(messages) > _LONG_HISTORY_TURNS
    many_tools = len(tools) > _MANY_TOOLS

    signals = {
        "long_prompt": long_prompt,
        "complex_keyword": complex_keyword,
        "long_history": long_history,
        "many_tools": many_tools,
    }
    score = sum(1 for v in signals.values() if v)

    # Domain-aware signals (Phase 10)
    if domain_routing and last_content:
        domain_boost = any(p.search(last_content) for p in _DOMAIN_BOOST_PATTERNS.values())
        simple_pattern = bool(_SIMPLE_PATTERN.match(last_content))
        signals["domain_boost"] = domain_boost
        signals["simple_pattern"] = simple_pattern
        if domain_boost:
            score += 2
        if simple_pattern:
            score = max(0, score - 1)

    return score, signals


def _tier_from_score(score: int) -> Tier:
    if score == 0:
        return Tier.FAST
    if score <= _STANDARD_MAX_SCORE:
        return Tier.STANDARD
    return Tier.POWERFUL


def _classify_complexity(messages: list[Message], tools: list[dict]) -> Tier:
    """Score message complexity and return the appropriate tier.

    Heuristic signals (each contributes +1 to score):
    - Last message content longer than _LONG_PROMPT_CHARS (500) chars
    - More than _LONG_HISTORY_TURNS (10) messages in history
    - Complex keyword in last message (architect, design, refactor, optimize, analyze, compare)
    - More than _MANY_TOOLS (8) tools loaded

    Score 0 → FAST, 1..2 → STANDARD, 3+ → POWERFUL.
    """
    score, _ = _score_complexity(messages, tools)
    return _tier_from_score(score)


_TECHNICAL_ERROR_SIGNALS = (
    "402",
    "429",
    "502",
    "503",
    "timeout",
    "connection",
    "rate limit",
    "http error",  # was "http" — too broad, matched URLs and unrelated errors
    "httpx",  # litellm raises httpx.HTTPError-derived exceptions
    # Ollama cloud session cap: surfaces inside APIConnectionError body.
    # Must trigger fallback, not crash.
    "session usage limit",
    "reached your session",
    # OpenRouter/Venice malformed response: litellm tries int(finish_reason)
    # where finish_reason='tool_use_failed'. Retryable on another tier.
    "tool_use_failed",
)

_RATE_LIMIT_SIGNALS = ("429", "rate limit", "too many requests", "session usage limit", "reached your session")

_RETRY_AFTER_RE = re.compile(r'"retry_after_seconds"\s*:\s*(\d+(?:\.\d+)?)')
# Groq uses English: "Please try again in 9 seconds" or "in 2m30.5s"
_GROQ_RETRY_RE = re.compile(r'try again in (\d+(?:\.\d+)?)\s*s(?:econds?)?', re.IGNORECASE)
_MAX_AUTO_RETRY_SECONDS = 60.0
_MAX_SAME_TIER_RETRY_SECONDS = 30.0


def _parse_retry_after(exc: Exception) -> float | None:
    """Extract retry_after_seconds from a rate-limit error, if present and short."""
    msg = str(exc)
    m = _RETRY_AFTER_RE.search(msg)
    if m:
        secs = float(m.group(1))
        return secs if secs <= _MAX_AUTO_RETRY_SECONDS else None
    # Groq plain-text format: "Please try again in 9 seconds"
    m = _GROQ_RETRY_RE.search(msg)
    if m:
        secs = float(m.group(1))
        return secs if secs <= _MAX_AUTO_RETRY_SECONDS else None
    return None

_FALLBACK_ORDER: list[str] = ["fast", "standard", "powerful"]


def _is_technical_error(exc: Exception) -> bool:
    """Return True if the exception is a retryable technical error."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _TECHNICAL_ERROR_SIGNALS)


def _is_rate_limit_error(exc: Exception) -> bool:
    """True for 429 / session-cap errors specifically (subset of technical errors)."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _RATE_LIMIT_SIGNALS)


def prefixed_model_id(provider: str, model: str) -> str:
    """Apply the LiteLLM ``<provider>/<model>`` prefix convention.

    Single source of truth for the prefixing rule, reused by
    :func:`build_litellm_provider` and by callers that need the litellm model id
    without building a provider (e.g. the vision sub-call in image_read).
    """
    if provider in ("ollama", "openrouter", "groq", "github_copilot"):
        return f"{provider}/{model}"
    return model


def build_litellm_provider(
    provider: str,
    model: str,
    api_base: str | None,
    *,
    prompt_cache: bool = True,
    max_retries: int = 3,
    retry_backoff: float = 2.0,
    request_timeout: float | None = 90.0,
) -> LiteLLMProvider:
    """Build a LiteLLMProvider, applying the LiteLLM provider-prefix convention.

    Used both internally by `RouterProvider` (to instantiate per-tier providers) and
    by the CLI's legacy single-provider path, to keep the prefixing rule in one place.
    """
    prefixed_model = prefixed_model_id(provider, model)
    return LiteLLMProvider(
        model=prefixed_model,
        api_base=api_base,
        prompt_cache=prompt_cache,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        request_timeout=request_timeout,
    )


class RouterProvider:
    """Routes LLM requests across fast/standard/powerful tiers with technical-error fallback.

    On `complete()`: classifies prompt complexity (or uses an override) to pick a starting
    tier, then attempts each tier in fallback order. Technical errors (HTTP 4xx/5xx,
    timeouts, connection issues) escalate to the next tier; non-technical errors propagate.

    `stream()` does NOT fallback (streaming a partial response and then switching providers
    would corrupt the output).
    """

    def __init__(self, config: RouterConfig, default_tier: Tier | None = None) -> None:
        self._config = config
        self.default_tier = default_tier
        self._providers: dict[Tier, LiteLLMProvider] = {}
        valid_tiers = {t.value for t in Tier}
        for tier_name, tier_cfg in config.tiers.items():
            if tier_name not in valid_tiers:
                _logger.warning(
                    "Unknown router tier %r in config; ignoring (valid: %s)",
                    tier_name,
                    sorted(valid_tiers),
                )
                continue
            tier = Tier(tier_name)
            self._providers[tier] = build_litellm_provider(
                tier_cfg.provider, tier_cfg.model, tier_cfg.api_base
            )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tier_override: Tier | None = None,
    ) -> LLMResponse:
        tools_list = tools or []
        score, signals = _score_complexity(messages, tools_list)
        if tier_override is not None:
            start_tier = tier_override
            reason = "override"
        elif self.default_tier is not None:
            start_tier = self.default_tier
            reason = "default"
        else:
            start_tier = _tier_from_score(score)
            reason = "classified"

        # Observability is fail-open: never let a logging error affect routing.
        with contextlib.suppress(Exception):
            _log.info(
                EventName.ROUTING_DECISION,
                chosen_tier=start_tier.value,
                score=score,
                signals=signals,
                reason=reason,
            )
        return await self._call_with_fallback(
            start_tier,
            messages,
            tools,
            temperature,
            max_tokens,
        )

    async def _call_with_fallback(
        self,
        start_tier: Tier,
        messages: list[Message],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            start_index = _FALLBACK_ORDER.index(start_tier.value)
        except ValueError:
            start_index = 0
        tiers_to_try = [
            Tier(t) for t in _FALLBACK_ORDER[start_index:] if Tier(t) in self._providers
        ]

        last_exc: Exception | None = None
        for idx, tier in enumerate(tiers_to_try):
            provider = self._providers[tier]
            next_tier_name = (
                tiers_to_try[idx + 1].value if idx + 1 < len(tiers_to_try) else None
            )
            # Bind ``tier`` into the structlog contextvar stack so nested
            # ``llm.complete`` events inherit it. ``bound_contextvars`` restores
            # the prior state on exit — no leak into unrelated call sites.
            with structlog.contextvars.bound_contextvars(tier=tier.value):
                try:
                    return await provider.complete(messages, tools, temperature, max_tokens)
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if _is_technical_error(exc):
                        # Short-window rate limit (e.g. Groq TPM): retry same
                        # tier after the wait rather than burning the next tier.
                        retry_after = _parse_retry_after(exc)
                        if retry_after is not None and retry_after <= _MAX_SAME_TIER_RETRY_SECONDS:
                            await asyncio.sleep(retry_after + 1)
                            with contextlib.suppress(Exception):
                                return await provider.complete(
                                    messages, tools, temperature, max_tokens
                                )
                        with contextlib.suppress(Exception):
                            _log.warning(
                                EventName.FALLBACK,
                                from_tier=tier.value,
                                to_tier=next_tier_name,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                        continue  # escalate to next tier
                    raise

        # All tiers exhausted. If the last failure is a short-window rate limit
        # (e.g. Venice/OpenRouter burst cap with retry_after < 60s), wait and
        # retry the last available tier once rather than giving up entirely.
        if last_exc is not None:
            retry_after = _parse_retry_after(last_exc)
            if retry_after is not None and tiers_to_try:
                last_tier = tiers_to_try[-1]
                await asyncio.sleep(retry_after + 1)
                with structlog.contextvars.bound_contextvars(tier=last_tier.value):
                    return await self._providers[last_tier].complete(
                        messages, tools, temperature, max_tokens
                    )
            raise last_exc
        msg = "No providers configured for routing"
        raise RuntimeError(msg)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        tier_override: Tier | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream with pre-first-chunk fallback across tiers.

        Fallback is safe here because ``LiteLLMProvider.stream()`` is an async
        generator: the upstream connection (``await completion(**kwargs)``) is
        established on the FIRST ``__anext__`` call. Any error before we yield
        a chunk to OUR caller means no partial output has been sent, so we can
        transparently retry or escalate without corrupting the stream.

        Once at least one chunk is yielded downstream we commit to the current
        provider — mid-stream switching would corrupt the response.
        """
        start_tier = tier_override or self.default_tier or _classify_complexity(
            messages, tools or []
        )
        try:
            start_index = _FALLBACK_ORDER.index(start_tier.value)
        except ValueError:
            start_index = 0
        tiers_to_try = [
            Tier(t) for t in _FALLBACK_ORDER[start_index:] if Tier(t) in self._providers
        ]

        last_exc: Exception | None = None
        for idx, tier in enumerate(tiers_to_try):
            provider = self._providers[tier]
            next_tier_name = (
                tiers_to_try[idx + 1].value if idx + 1 < len(tiers_to_try) else None
            )
            with structlog.contextvars.bound_contextvars(tier=tier.value):
                yielded = False
                try:
                    async for chunk in provider.stream(messages, tools, temperature, max_tokens):
                        yielded = True
                        yield chunk
                    return  # stream completed successfully
                except Exception as exc:  # noqa: BLE001
                    if yielded:
                        raise  # mid-stream: cannot recover, propagate
                    last_exc = exc
                    if _is_technical_error(exc):
                        # Short-window rate limit: retry same tier before escalating.
                        retry_after = _parse_retry_after(exc)
                        if retry_after is not None and retry_after <= _MAX_SAME_TIER_RETRY_SECONDS:
                            await asyncio.sleep(retry_after + 1)
                            try:
                                async for chunk in provider.stream(
                                    messages, tools, temperature, max_tokens
                                ):
                                    yielded = True
                                    yield chunk
                                return
                            except Exception:  # noqa: BLE001
                                pass  # retry failed → fall through to escalate
                        with contextlib.suppress(Exception):
                            _log.warning(
                                EventName.FALLBACK,
                                from_tier=tier.value,
                                to_tier=next_tier_name,
                                error_type=type(exc).__name__,
                                error_message=str(exc),
                            )
                        continue  # escalate to next tier
                    raise  # non-technical: propagate immediately

        # All tiers exhausted. If the last failure is a short-window rate limit,
        # wait and retry the last available tier once (streaming path).
        if last_exc is not None:
            retry_after = _parse_retry_after(last_exc)
            if retry_after is not None and tiers_to_try:
                last_tier = tiers_to_try[-1]
                await asyncio.sleep(retry_after + 1)
                with structlog.contextvars.bound_contextvars(tier=last_tier.value):
                    async for chunk in self._providers[last_tier].stream(
                        messages, tools, temperature, max_tokens
                    ):
                        yield chunk
                    return
            raise last_exc
        msg = "No providers configured for routing"
        raise RuntimeError(msg)
