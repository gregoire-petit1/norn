# Phase 7: RouterProvider Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `RouterProvider` that routes requests to fast/standard/powerful tiers based on prompt complexity, with automatic fallback on technical errors and a `--model` CLI override.

**Architecture:** A new `RouterProvider` class in `src/norn/core/router.py` implements the `LLMProvider` protocol (same interface as `LiteLLMProvider`). A local heuristic scores message complexity to select a tier. On technical error (HTTP, timeout, 402, connection), it escalates to the next tier. `_build_provider()` in `main.py` returns a `RouterProvider` when `config.router.enabled = True`, otherwise the existing `LiteLLMProvider` unchanged.

**Tech Stack:** Python 3.11+, Pydantic v2, litellm, typer, pytest, ruff. Uses existing `LiteLLMProvider`, `LLMProvider` protocol, `NornConfig`, `Message`, `LLMResponse`, `StreamChunk`.

---

### Task 1: RouterConfig in config.py

**Files:**
- Modify: `src/norn/core/config.py`
- Test: `tests/test_core/test_config.py`

**Step 1: Write the failing tests**

Add to `tests/test_core/test_config.py`:

```python
from norn.core.config import RouterConfig, RouterTierConfig

def test_router_config_defaults():
    config = NornConfig()
    assert config.router.enabled is False
    assert config.router.tiers == {}

def test_router_tier_config():
    tier = RouterTierConfig(provider="ollama", model="qwen2.5-coder:7b")
    assert tier.provider == "ollama"
    assert tier.model == "qwen2.5-coder:7b"
    assert tier.api_base is None

def test_router_config_with_tiers():
    config = NornConfig(
        router={
            "enabled": True,
            "tiers": {
                "fast": {"provider": "ollama", "model": "qwen2.5-coder:7b"},
                "standard": {"provider": "openrouter", "model": "stepfun/step-3.5-flash:free"},
                "powerful": {"provider": "openrouter", "model": "anthropic/claude-sonnet-4"},
            },
        }
    )
    assert config.router.enabled is True
    assert config.router.tiers["fast"].model == "qwen2.5-coder:7b"
    assert config.router.tiers["standard"].provider == "openrouter"
    assert config.router.tiers["powerful"].model == "anthropic/claude-sonnet-4"

def test_router_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "router:\n"
        "  enabled: true\n"
        "  tiers:\n"
        "    fast:\n"
        "      provider: ollama\n"
        "      model: qwen2.5-coder:7b\n"
    )
    config = NornConfig.from_yaml(config_file)
    assert config.router.enabled is True
    assert config.router.tiers["fast"].provider == "ollama"
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/norn && uv run pytest tests/test_core/test_config.py -k "router" -v
```

Expected: FAIL with `ImportError: cannot import name 'RouterConfig'`

**Step 3: Implement RouterTierConfig and RouterConfig in config.py**

Add after the `CoordinatorConfig` class (before `NornConfig`), and add the `router` field to `NornConfig`:

```python
class RouterTierConfig(BaseModel):
    provider: str
    model: str
    api_base: str | None = None


class RouterConfig(BaseModel):
    enabled: bool = False
    tiers: dict[str, RouterTierConfig] = {}
```

Then in `NornConfig`, add the field:
```python
class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
    mcp: MCPConfig = MCPConfig()
    router: RouterConfig = RouterConfig()  # ADD THIS
```

Also update the import in test file: add `RouterConfig, RouterTierConfig` to the import from `norn.core.config`.

**Step 4: Run tests to verify they pass**

```bash
cd ~/norn && uv run pytest tests/test_core/test_config.py -k "router" -v
```

Expected: 4 tests PASS

**Step 5: Run ruff**

```bash
cd ~/norn && uv run ruff check src/norn/core/config.py
```

Expected: no errors

**Step 6: Commit**

```bash
cd ~/norn && git add src/norn/core/config.py tests/test_core/test_config.py
git commit -m "feat(router): add RouterConfig and RouterTierConfig to NornConfig"
```

---

### Task 2: Complexity classifier (_classify_complexity)

**Files:**
- Create: `src/norn/core/router.py`
- Create: `tests/test_core/test_router.py`

**Scoring table:**

| Condition | +score |
|-----------|--------|
| `len(last_message.content) > 500` | +1 |
| `len(messages) > 10` | +1 |
| Any keyword in last message: `architect`, `design`, `refactor`, `optimize`, `analyze`, `compare` | +1 |
| `len(tools) > 8` | +1 |

Score 0 → `Tier.FAST`, 1–2 → `Tier.STANDARD`, 3+ → `Tier.POWERFUL`

**Step 1: Write the failing tests**

Create `tests/test_core/test_router.py`:

```python
"""Tests for RouterProvider."""

from __future__ import annotations

import pytest

from norn.core.models import Message, Role
from norn.core.router import Tier, _classify_complexity


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


def test_classify_keyword_in_prompt_is_standard():
    for kw in ("architect", "design", "refactor", "optimize", "analyze", "compare"):
        messages = [_msg(f"please {kw} this")]
        assert _classify_complexity(messages, []) == Tier.STANDARD, f"keyword={kw}"


def test_classify_keyword_case_insensitive():
    messages = [_msg("REFACTOR this module")]
    assert _classify_complexity(messages, []) == Tier.STANDARD


def test_classify_many_tools_is_standard():
    tools = [{"name": f"tool_{i}"} for i in range(9)]
    messages = [_msg("hello")]
    assert _classify_complexity(messages, tools) == Tier.STANDARD


def test_classify_multiple_signals_is_powerful():
    # long prompt + keyword = score 2 → STANDARD
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
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'norn.core.router'`

**Step 3: Create src/norn/core/router.py with Tier enum and _classify_complexity**

```python
"""RouterProvider — 3-tier LLM routing with complexity-based selection and fallback."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from norn.core.models import Message, StreamChunk


_COMPLEX_KEYWORDS = frozenset(
    {"architect", "design", "refactor", "optimize", "analyze", "compare"}
)


class Tier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


def _classify_complexity(messages: list[Message], tools: list[dict]) -> Tier:
    """Score message complexity and return the appropriate tier."""
    score = 0

    # Signal 1: long last message
    if messages:
        last = messages[-1]
        if last.content and len(last.content) > 500:
            score += 1

    # Signal 2: long conversation history
    if len(messages) > 10:
        score += 1

    # Signal 3: complex keywords in last message
    if messages:
        last = messages[-1]
        if last.content:
            words = last.content.lower().split()
            if _COMPLEX_KEYWORDS.intersection(words):
                score += 1

    # Signal 4: many tools loaded
    if len(tools) > 8:
        score += 1

    if score == 0:
        return Tier.FAST
    if score <= 2:
        return Tier.STANDARD
    return Tier.POWERFUL
```

**Step 4: Run tests to verify they pass**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py -v
```

Expected: all tests PASS

**Step 5: Run ruff**

```bash
cd ~/norn && uv run ruff check src/norn/core/router.py
```

Expected: no errors

**Step 6: Commit**

```bash
cd ~/norn && git add src/norn/core/router.py tests/test_core/test_router.py
git commit -m "feat(router): add Tier enum and complexity classifier"
```

---

### Task 3: RouterProvider.complete() with fallback

**Files:**
- Modify: `src/norn/core/router.py`
- Modify: `tests/test_core/test_router.py`

**Design notes:**
- `RouterProvider.__init__` takes a `RouterConfig` and an optional `api_key` str.
- It builds a `LiteLLMProvider` for each configured tier on init.
- `_FALLBACK_ORDER = [Tier.FAST, Tier.STANDARD, Tier.POWERFUL]`
- `complete()` calls `_call_with_fallback(tier, ...)`.
- `_call_with_fallback()`: try the tier; on `Exception` containing HTTP/timeout/402/connection indicators, escalate to next tier; if no next tier, re-raise.
- Technical error detection: catch all `Exception`, then check message for: `"402"`, `"timeout"`, `"connection"`, `"http"`, `"rate limit"`, `"503"`, `"502"`, `"429"`. If none match, re-raise immediately (don't fallback on logic/validation errors).

**Step 1: Write the failing tests**

Add to `tests/test_core/test_router.py`:

```python
from unittest.mock import AsyncMock, patch

from norn.core.config import RouterConfig, RouterTierConfig
from norn.core.models import LLMResponse, TokenUsage
from norn.core.router import RouterProvider


def _make_router_config() -> RouterConfig:
    return RouterConfig(
        enabled=True,
        tiers={
            "fast": RouterTierConfig(provider="ollama", model="qwen2.5-coder:7b"),
            "standard": RouterTierConfig(provider="openrouter", model="stepfun/step-3.5-flash:free"),
            "powerful": RouterTierConfig(provider="openrouter", model="anthropic/claude-sonnet-4"),
        },
    )


def _make_response(content: str = "ok") -> LLMResponse:
    return LLMResponse(content=content, usage=TokenUsage())


@pytest.mark.asyncio
async def test_router_complete_uses_fast_tier_for_simple_prompt():
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.FAST], "complete", new_callable=AsyncMock,
        return_value=_make_response("fast response")
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
        router._providers[Tier.POWERFUL], "complete", new_callable=AsyncMock,
        return_value=_make_response("powerful response")
    ) as mock_powerful:
        result = await router.complete(messages)
        assert result.content == "powerful response"
        mock_powerful.assert_called_once()


@pytest.mark.asyncio
async def test_router_fallback_on_http_error():
    config = _make_router_config()
    router = RouterProvider(config)

    # fast tier raises 402; standard succeeds
    with (
        patch.object(
            router._providers[Tier.FAST], "complete", new_callable=AsyncMock,
            side_effect=Exception("402 Payment Required")
        ),
        patch.object(
            router._providers[Tier.STANDARD], "complete", new_callable=AsyncMock,
            return_value=_make_response("standard response")
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
            router._providers[Tier.FAST], "complete", new_callable=AsyncMock,
            side_effect=Exception("connection timeout")
        ),
        patch.object(
            router._providers[Tier.STANDARD], "complete", new_callable=AsyncMock,
            return_value=_make_response("ok")
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
        router._providers[Tier.FAST], "complete", new_callable=AsyncMock,
        side_effect=ValueError("invalid argument")
    ):
        with pytest.raises(ValueError, match="invalid argument"):
            await router.complete([_msg("hello")])


@pytest.mark.asyncio
async def test_router_raises_when_all_tiers_exhausted():
    """If all tiers fail with technical errors, re-raise the last error."""
    config = _make_router_config()
    router = RouterProvider(config)

    with (
        patch.object(
            router._providers[Tier.FAST], "complete", new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway")
        ),
        patch.object(
            router._providers[Tier.STANDARD], "complete", new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway")
        ),
        patch.object(
            router._providers[Tier.POWERFUL], "complete", new_callable=AsyncMock,
            side_effect=Exception("502 Bad Gateway")
        ),
    ):
        with pytest.raises(Exception, match="502"):
            await router.complete([_msg("hello")])


@pytest.mark.asyncio
async def test_router_complete_with_tier_override():
    """When tier_override is given, skip classification and use that tier."""
    config = _make_router_config()
    router = RouterProvider(config)

    with patch.object(
        router._providers[Tier.POWERFUL], "complete", new_callable=AsyncMock,
        return_value=_make_response("powerful override")
    ) as mock_powerful:
        result = await router.complete([_msg("hello")], tier_override=Tier.POWERFUL)
        assert result.content == "powerful override"
        mock_powerful.assert_called_once()
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py -k "router_complete or router_fallback or router_raises or router_no_fallback or tier_override" -v
```

Expected: FAIL with `ImportError` or `AttributeError`

**Step 3: Implement RouterProvider in router.py**

Add the full class after `_classify_complexity`. The complete `router.py` should be:

```python
"""RouterProvider — 3-tier LLM routing with complexity-based selection and fallback."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

from norn.core.config import RouterConfig
from norn.core.llm import LiteLLMProvider
from norn.core.models import LLMResponse, Message, StreamChunk


_COMPLEX_KEYWORDS = frozenset(
    {"architect", "design", "refactor", "optimize", "analyze", "compare"}
)

_TECHNICAL_ERROR_SIGNALS = ("402", "timeout", "connection", "http", "rate limit", "503", "502", "429")

_FALLBACK_ORDER = ["fast", "standard", "powerful"]


class Tier(StrEnum):
    FAST = "fast"
    STANDARD = "standard"
    POWERFUL = "powerful"


def _classify_complexity(messages: list[Message], tools: list[dict]) -> Tier:
    """Score message complexity and return the appropriate tier."""
    score = 0

    if messages:
        last = messages[-1]
        if last.content and len(last.content) > 500:
            score += 1

    if len(messages) > 10:
        score += 1

    if messages:
        last = messages[-1]
        if last.content:
            words = last.content.lower().split()
            if _COMPLEX_KEYWORDS.intersection(words):
                score += 1

    if len(tools) > 8:
        score += 1

    if score == 0:
        return Tier.FAST
    if score <= 2:
        return Tier.STANDARD
    return Tier.POWERFUL


def _is_technical_error(exc: Exception) -> bool:
    """Return True if the exception is a retryable technical error."""
    msg = str(exc).lower()
    return any(signal in msg for signal in _TECHNICAL_ERROR_SIGNALS)


def _build_tier_provider(provider: str, model: str, api_base: str | None) -> LiteLLMProvider:
    """Build a LiteLLMProvider for a single tier."""
    prefixed_model = model
    if provider == "ollama":
        prefixed_model = f"ollama/{model}"
    elif provider == "openrouter":
        prefixed_model = f"openrouter/{model}"
    return LiteLLMProvider(model=prefixed_model, api_base=api_base)


class RouterProvider:
    """Routes LLM requests across fast/standard/powerful tiers with fallback."""

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._providers: dict[Tier, LiteLLMProvider] = {}
        for tier_name, tier_cfg in config.tiers.items():
            tier = Tier(tier_name)
            self._providers[tier] = _build_tier_provider(
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
        start_tier = tier_override or _classify_complexity(messages, tools or [])
        return await self._call_with_fallback(start_tier, messages, tools, temperature, max_tokens)

    async def _call_with_fallback(
        self,
        start_tier: Tier,
        messages: list[Message],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        # Build the ordered list of tiers to try, starting from start_tier
        try:
            start_index = _FALLBACK_ORDER.index(start_tier.value)
        except ValueError:
            start_index = 0
        tiers_to_try = [Tier(t) for t in _FALLBACK_ORDER[start_index:] if Tier(t) in self._providers]

        last_exc: Exception | None = None
        for tier in tiers_to_try:
            provider = self._providers[tier]
            try:
                return await provider.complete(messages, tools, temperature, max_tokens)
            except Exception as exc:
                if _is_technical_error(exc):
                    last_exc = exc
                    continue
                raise  # Non-technical error: propagate immediately

        if last_exc is not None:
            raise last_exc
        msg = "No providers available for routing"
        raise RuntimeError(msg)

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        start_tier = _classify_complexity(messages, tools or [])
        provider = self._providers.get(start_tier)
        if provider is None:
            msg = f"No provider configured for tier {start_tier}"
            raise RuntimeError(msg)
        return await provider.stream(messages, tools, temperature, max_tokens)
```

**Step 4: Run tests to verify they pass**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py -v
```

Expected: all tests PASS

**Step 5: Run ruff**

```bash
cd ~/norn && uv run ruff check src/norn/core/router.py
```

Expected: no errors

**Step 6: Commit**

```bash
cd ~/norn && git add src/norn/core/router.py tests/test_core/test_router.py
git commit -m "feat(router): implement RouterProvider with 3-tier routing and technical error fallback"
```

---

### Task 4: Wire RouterProvider into _build_provider() in main.py

**Files:**
- Modify: `src/norn/cli/main.py`

**Design:** `_build_provider()` signature changes to accept an optional `tier_override: Tier | None = None`. When `config.router.enabled = True`, it returns a `RouterProvider`; otherwise, the existing `LiteLLMProvider`. The `tier_override` is stored on the `RouterProvider` instance so `chat`/`run`/etc. can pass it through at call time.

Actually, since `AgentLoop.run()` doesn't accept a tier, the CLI `--model` override must be plumbed differently. The cleanest approach: **store `tier_override` on the `RouterProvider` instance as a default**, overriding what `_classify_complexity` would return.

**Step 1: Write the failing test**

Add to `tests/test_core/test_router.py`:

```python
def test_router_provider_default_tier_override():
    """RouterProvider.default_tier overrides classification when set."""
    config = _make_router_config()
    router = RouterProvider(config, default_tier=Tier.POWERFUL)
    assert router.default_tier == Tier.POWERFUL
```

**Step 2: Run to verify it fails**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py::test_router_provider_default_tier_override -v
```

Expected: FAIL with `TypeError`

**Step 3: Add `default_tier` to RouterProvider.__init__ and complete()**

In `src/norn/core/router.py`, update `RouterProvider`:

```python
class RouterProvider:
    def __init__(self, config: RouterConfig, default_tier: Tier | None = None) -> None:
        self._config = config
        self.default_tier = default_tier
        self._providers: dict[Tier, LiteLLMProvider] = {}
        for tier_name, tier_cfg in config.tiers.items():
            tier = Tier(tier_name)
            self._providers[tier] = _build_tier_provider(
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
        start_tier = tier_override or self.default_tier or _classify_complexity(messages, tools or [])
        return await self._call_with_fallback(start_tier, messages, tools, temperature, max_tokens)
```

**Step 4: Update _build_provider() in main.py**

Change signature and body:

```python
# Add import at top of file (with other TYPE_CHECKING imports or direct):
from norn.core.router import RouterProvider, Tier

def _build_provider(config: NornConfig, model_override: str | None = None) -> LiteLLMProvider | RouterProvider:
    """Build the LLM provider from config."""
    if config.router.enabled:
        default_tier: Tier | None = None
        if model_override is not None:
            try:
                default_tier = Tier(model_override)
            except ValueError:
                console.print(f"[yellow]Unknown tier '{model_override}'. Valid: fast, standard, powerful. Ignoring.[/yellow]")
        return RouterProvider(config.router, default_tier=default_tier)
    # Legacy path
    model = config.llm.model
    if config.llm.provider == "ollama":
        model = f"ollama/{config.llm.model}"
    elif config.llm.provider == "openrouter":
        model = f"openrouter/{config.llm.model}"
    return LiteLLMProvider(model=model, api_base=config.llm.api_base)
```

**Step 5: Run tests to verify they pass**

```bash
cd ~/norn && uv run pytest tests/test_core/test_router.py -v
```

Expected: all tests PASS

**Step 6: Run ruff on both files**

```bash
cd ~/norn && uv run ruff check src/norn/core/router.py src/norn/cli/main.py
```

Expected: no new errors (existing E402 warnings in main.py are accepted)

**Step 7: Commit**

```bash
cd ~/norn && git add src/norn/core/router.py src/norn/cli/main.py
git commit -m "feat(router): wire RouterProvider into _build_provider with default_tier support"
```

---

### Task 5: Add --model CLI option to chat, run, dream, coordinate

**Files:**
- Modify: `src/norn/cli/main.py`

**Step 1: Write the failing tests**

Add to a new file `tests/test_cli_router.py`:

```python
"""Tests for --model CLI option wiring."""

from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from norn.cli.main import app

runner = CliRunner()


def test_chat_accepts_model_option():
    """--model fast should not cause a CLI error (option exists)."""
    with patch("norn.cli.main._build_provider") as mock_build:
        mock_build.return_value = MagicMock()
        # We just test the option is accepted by typer (no "No such option" error)
        result = runner.invoke(app, ["chat", "--model", "fast", "--help"])
        assert "--model" in result.output


def test_run_accepts_model_option():
    result = runner.invoke(app, ["run", "--help"])
    assert "--model" in result.output


def test_dream_accepts_model_option():
    result = runner.invoke(app, ["dream", "--help"])
    assert "--model" in result.output


def test_coordinate_accepts_model_option():
    result = runner.invoke(app, ["coordinate", "--help"])
    assert "--model" in result.output
```

**Step 2: Run tests to verify they fail**

```bash
cd ~/norn && uv run pytest tests/test_cli_router.py -v
```

Expected: FAIL (option not yet in help text)

**Step 3: Add --model option to all four commands**

For each command (`chat`, `run`, `dream`, `coordinate`), add a `model` param and pass it to `_build_provider`. Pattern for each:

```python
@app.command()
def chat(
    model: str | None = typer.Option(None, "--model", help="Force routing tier: fast, standard, or powerful"),
) -> None:
    ...
    provider = _build_provider(config, model_override=model)
    ...
```

The `run` command already has `prompt` as an `Argument`; add `model` as an `Option`:

```python
@app.command()
def run(
    prompt: str = typer.Argument(help="One-shot prompt to execute"),
    model: str | None = typer.Option(None, "--model", help="Force routing tier: fast, standard, or powerful"),
) -> None:
```

Apply the same pattern to `dream` and `coordinate`.

**Step 4: Run tests to verify they pass**

```bash
cd ~/norn && uv run pytest tests/test_cli_router.py -v
```

Expected: all PASS

**Step 5: Run ruff**

```bash
cd ~/norn && uv run ruff check src/norn/cli/main.py
```

Expected: no new errors

**Step 6: Commit**

```bash
cd ~/norn && git add src/norn/cli/main.py tests/test_cli_router.py
git commit -m "feat(router): add --model CLI option to chat, run, dream, coordinate commands"
```

---

### Task 6: Update configs/default.yaml with router block

**Files:**
- Modify: `configs/default.yaml`

**Step 1: Add the router block**

Add at the end of `configs/default.yaml`:

```yaml
router:
  enabled: false        # Phase 7: 3-tier routing (fast/standard/powerful)
  tiers:
    fast:
      provider: "ollama"
      model: "qwen2.5-coder:7b"
    standard:
      provider: "openrouter"
      model: "stepfun/step-3.5-flash:free"
    powerful:
      provider: "openrouter"
      model: "anthropic/claude-sonnet-4"
```

**Step 2: Verify config loads cleanly**

```bash
cd ~/norn && uv run python -c "from norn.core.config import NornConfig; c = NornConfig.load(); print(c.router)"
```

Expected: prints `RouterConfig(enabled=False, tiers={...})` with 3 tiers

**Step 3: Commit**

```bash
cd ~/norn && git add configs/default.yaml
git commit -m "chore: add router block to default.yaml"
```

---

### Task 7: Full test suite verification

**Step 1: Run the full test suite**

```bash
cd ~/norn && uv run pytest --tb=short -q
```

Expected: all 354+ tests PASS (plus new router/cli tests), 0 failures

**Step 2: Run ruff on the full project**

```bash
cd ~/norn && uv run ruff check src/ tests/
```

Expected: no new errors (existing E402 in main.py are accepted)

**Step 3: Final commit if any cleanup needed**

```bash
cd ~/norn && git add -A
git commit -m "chore: phase 7 complete — RouterProvider with 3-tier routing"
```

---

## Summary

| Task | Files | Commits |
|------|-------|---------|
| 1. RouterConfig | `config.py`, `test_config.py` | 1 |
| 2. Complexity classifier | `router.py` (new), `test_router.py` (new) | 1 |
| 3. RouterProvider.complete() + fallback | `router.py`, `test_router.py` | 1 |
| 4. Wire into _build_provider | `router.py`, `main.py` | 1 |
| 5. --model CLI option | `main.py`, `test_cli_router.py` | 1 |
| 6. default.yaml router block | `configs/default.yaml` | 1 |
| 7. Full suite verification | — | 0–1 |

**Total: 6–7 commits, ~40 new tests**
