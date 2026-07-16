"""Tests for LLM provider abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norn.core.llm import LiteLLMProvider, build_tool_schemas
from norn.core.models import LLMResponse, Message, Role


@pytest.fixture
def provider():
    return LiteLLMProvider(model="ollama/qwen2.5-coder:14b")


def test_provider_creation(provider):
    assert provider.model == "ollama/qwen2.5-coder:14b"


def test_build_tool_schemas_empty():
    schemas = build_tool_schemas([])
    assert schemas == []


def test_build_tool_schemas_from_dict():
    tool_def = {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    schemas = build_tool_schemas([tool_def])
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_complete_text_response(provider):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(messages=[Message(role=Role.USER, content="Hi")])
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.has_tool_calls is False


@pytest.mark.asyncio
async def test_complete_with_tool_calls(provider):
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc"
    mock_tool_call.function.name = "bash"
    mock_tool_call.function.arguments = '{"command": "ls"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 30

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(
            messages=[Message(role=Role.USER, content="list files")],
            tools=[
                {
                    "name": "bash",
                    "description": "Run shell",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
        )
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}


# --------------------------------------------------------------------------- #
# D3.4 - empty messages edge case
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_complete_empty_messages_raises_value_error(provider):
    """Empty ``messages`` is a programmer error: a coding agent must always
    have at least a system or user turn to react to. Without an explicit
    guard, behaviour depends on the underlying provider — some 4xx, some
    silently return empty completions, some hang. Surface it early and
    cheaply with a deterministic ``ValueError``, before any network call.

    The acompletion mock must NEVER be awaited — the guard short-circuits.
    """
    fake_completion = AsyncMock()
    p = LiteLLMProvider(model="ollama/qwen2.5-coder:14b", completion_fn=fake_completion)

    with pytest.raises(ValueError, match="messages cannot be empty"):
        await p.complete(messages=[])

    fake_completion.assert_not_awaited()


# --------------------------------------------------------------------------- #
# Streaming tests
# --------------------------------------------------------------------------- #


def _make_stream_chunk(content=None, tool_calls=None, finish_reason=None, usage=None):
    """Build a mock litellm streaming chunk."""
    delta = MagicMock()
    delta.content = content
    delta.tool_calls = tool_calls
    choice = MagicMock()
    choice.delta = delta
    choice.finish_reason = finish_reason
    chunk = MagicMock()
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


async def _mock_stream_response(chunks):
    """Create an async iterable from a list of mock chunks."""
    for c in chunks:
        yield c


@pytest.mark.asyncio
async def test_stream_yields_text_chunks():
    chunks = [
        _make_stream_chunk(content="Hello"),
        _make_stream_chunk(content=" world"),
        _make_stream_chunk(content=None, finish_reason="stop"),
    ]

    async def mock_completion(**kwargs):
        return _mock_stream_response(chunks)

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="hi")]

    collected = []
    async for chunk in provider.stream(messages):
        collected.append(chunk)

    assert any(c.content == "Hello" for c in collected)
    assert any(c.content == " world" for c in collected)
    assert collected[-1].done is True


@pytest.mark.asyncio
async def test_stream_accumulates_tool_calls():
    """Tool call fragments across chunks should be assembled into complete ToolCalls."""
    tc_frag_1 = MagicMock()
    tc_frag_1.index = 0
    tc_frag_1.id = "call_abc"
    tc_frag_1.function = MagicMock()
    tc_frag_1.function.name = "bash"
    tc_frag_1.function.arguments = '{"comm'

    tc_frag_2 = MagicMock()
    tc_frag_2.index = 0
    tc_frag_2.id = None
    tc_frag_2.function = MagicMock()
    tc_frag_2.function.name = None
    tc_frag_2.function.arguments = 'and": "ls"}'

    chunks = [
        _make_stream_chunk(tool_calls=[tc_frag_1]),
        _make_stream_chunk(tool_calls=[tc_frag_2]),
        _make_stream_chunk(finish_reason="tool_calls"),
    ]

    async def mock_completion(**kwargs):
        return _mock_stream_response(chunks)

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="run ls")]

    collected = []
    async for chunk in provider.stream(messages):
        collected.append(chunk)

    final = collected[-1]
    assert final.done is True
    assert final.tool_calls is not None
    assert len(final.tool_calls) == 1
    assert final.tool_calls[0].name == "bash"
    assert final.tool_calls[0].arguments == {"command": "ls"}


@pytest.mark.asyncio
async def test_stream_uses_completion_fn():
    """stream() should use self._completion_fn, not hardcoded litellm."""
    called_with = {}

    async def mock_completion(**kwargs):
        called_with.update(kwargs)
        return _mock_stream_response(
            [
                _make_stream_chunk(content="ok", finish_reason="stop"),
            ]
        )

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)
    messages = [Message(role=Role.USER, content="hi")]

    async for _ in provider.stream(messages):
        pass

    assert called_with["stream"] is True
    assert called_with["model"] == "test/model"


@pytest.mark.asyncio
async def test_stream_empty_messages_raises():
    async def mock_completion(**kwargs):
        return _mock_stream_response([])

    provider = LiteLLMProvider("test/model", completion_fn=mock_completion)

    with pytest.raises(ValueError, match="messages cannot be empty"):
        async for _ in provider.stream(messages=[]):
            pass


@pytest.mark.asyncio
async def test_complete_request_timeout_passed_to_litellm():
    """The provider must forward ``request_timeout`` as ``timeout=`` so
    litellm gives up on stalled upstream providers.
    """
    called_with: dict = {}

    async def mock_completion(**kwargs):
        called_with.update(kwargs)
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "ok"
        mock_response.choices[0].message.tool_calls = None
        mock_response.usage.prompt_tokens = 1
        mock_response.usage.completion_tokens = 1
        mock_response.usage.total_tokens = 2
        return mock_response

    provider = LiteLLMProvider(
        "test/model", completion_fn=mock_completion, request_timeout=12.5
    )
    await provider.complete(messages=[Message(role=Role.USER, content="hi")])
    assert called_with.get("timeout") == 12.5


@pytest.mark.asyncio
async def test_complete_raises_on_provider_hang():
    """If the upstream call hangs beyond ``request_timeout`` we must raise.

    Reproduces ml-tools-001 silent stall: previously the LLM call never
    returned and the entire bench task hit its 180s wall-clock timeout
    with no stdout/stderr. With a request timeout, the provider raises
    ``TimeoutError`` and the agent loop can surface it.
    """
    import asyncio

    async def hang(**kwargs):
        await asyncio.sleep(10)
        raise AssertionError("should have been cancelled")

    provider = LiteLLMProvider(
        "test/model", completion_fn=hang, request_timeout=0.05
    )
    with pytest.raises(asyncio.TimeoutError):
        await provider.complete(messages=[Message(role=Role.USER, content="hi")])


# --------------------------------------------------------------------------- #
# Transient-error retry (rate limit + connection/5xx faults)
# --------------------------------------------------------------------------- #

from norn.core.llm import _is_transient_error, _retry_on_transient_error


class _StatusError(Exception):
    def __init__(self, status_code):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class TestIsTransientError:
    def test_429_status(self):
        assert _is_transient_error(_StatusError(429)) is True

    def test_500_status(self):
        assert _is_transient_error(_StatusError(503)) is True

    def test_400_not_transient(self):
        assert _is_transient_error(_StatusError(400)) is False

    def test_connection_error_by_message(self):
        assert _is_transient_error(Exception("APIConnectionError: Connection error")) is True

    def test_internal_server_error_by_message(self):
        assert _is_transient_error(Exception("InternalServerError: Overloaded")) is True

    def test_rate_limit_by_message(self):
        assert _is_transient_error(Exception("429 too many requests")) is True

    def test_context_window_not_transient(self):
        assert _is_transient_error(Exception("context window exceeded")) is False

    def test_auth_error_not_transient(self):
        assert _is_transient_error(_StatusError(401)) is False


@pytest.mark.asyncio
async def test_retry_succeeds_after_transient_failures():
    """A connection error should be retried and eventually succeed."""
    calls = {"n": 0}

    async def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise Exception("APIConnectionError: Connection error")
        return "ok"

    with patch("norn.core.llm.asyncio.sleep", new=AsyncMock()):
        result = await _retry_on_transient_error(flaky, max_retries=3, backoff_base=0.0)

    assert result == "ok"
    assert calls["n"] == 3


@pytest.mark.asyncio
async def test_retry_exhausts_and_raises():
    """Persistent transient error raises after exhausting retries."""
    calls = {"n": 0}

    async def always_fail():
        calls["n"] += 1
        raise Exception("InternalServerError: Overloaded")

    with patch("norn.core.llm.asyncio.sleep", new=AsyncMock()):
        with pytest.raises(Exception, match="Overloaded"):
            await _retry_on_transient_error(always_fail, max_retries=2, backoff_base=0.0)

    assert calls["n"] == 3  # initial + 2 retries


@pytest.mark.asyncio
async def test_non_transient_error_not_retried():
    """A 400 client error must fail immediately without retry."""
    calls = {"n": 0}

    async def bad_request():
        calls["n"] += 1
        raise _StatusError(400)

    with pytest.raises(_StatusError):
        await _retry_on_transient_error(bad_request, max_retries=3, backoff_base=0.0)

    assert calls["n"] == 1
