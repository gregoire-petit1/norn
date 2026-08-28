"""Tests for record/replay of LLM exchanges (SOTA v2, workstream A)."""

import json

import pytest

from norn.core.models import LLMResponse, Message, Role, StreamChunk, TokenUsage, ToolCall
from norn.core.replay import (
    RecordingProvider,
    ReplayDriftError,
    ReplayProvider,
    canonical_request,
    request_hash,
)


class _StubProvider:
    """Deterministic inner provider for recording tests."""

    model = "stub/model-1"

    def __init__(self, responses: list[LLMResponse]) -> None:
        self._responses = list(responses)

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return self._responses.pop(0)

    async def stream(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        response = self._responses.pop(0)
        content = response.content or ""
        for i in range(0, len(content), 3):
            yield StreamChunk(content=content[i : i + 3], done=False)
        yield StreamChunk(
            content=None,
            tool_calls=response.tool_calls or None,
            done=True,
            usage=response.usage,
        )


def _messages() -> list[Message]:
    return [
        Message(role=Role.SYSTEM, content="sys"),
        Message(role=Role.USER, content="hello"),
    ]


# --------------------------------------------------------------------------- #
# Canonical request hashing
# --------------------------------------------------------------------------- #


def test_request_hash_stable_across_key_order():
    msgs = _messages()
    tools_a = [{"name": "bash", "description": "d", "parameters": {}}]
    tools_b = [{"parameters": {}, "description": "d", "name": "bash"}]
    h1 = request_hash(canonical_request(msgs, tools_a, 0.0, 100))
    h2 = request_hash(canonical_request(msgs, tools_b, 0.0, 100))
    assert h1 == h2


def test_request_hash_sensitive_to_content():
    msgs = _messages()
    h1 = request_hash(canonical_request(msgs, None, 0.0, 100))
    h2 = request_hash(canonical_request(msgs, None, 0.5, 100))
    assert h1 != h2


# --------------------------------------------------------------------------- #
# Round-trip: record then replay
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_record_then_replay_round_trip(tmp_path):
    path = tmp_path / "session.jsonl"
    recorded_response = LLMResponse(
        content="hi there",
        tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "x"})],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="stub/model-1",
    )
    recorder = RecordingProvider(_StubProvider([recorded_response]), path)
    live = await recorder.complete(messages=_messages())
    assert live.content == "hi there"

    # File: meta line + one exchange line
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["kind"] == "meta"
    assert lines[1]["kind"] == "exchange"
    assert lines[1]["seq"] == 0

    replay = ReplayProvider(path)
    replayed = await replay.complete(messages=_messages())
    assert replayed.content == "hi there"
    assert replayed.tool_calls[0].name == "echo"
    assert replayed.usage.prompt_tokens == 10
    replay.assert_exhausted()


@pytest.mark.asyncio
async def test_replay_drift_raises_with_diff(tmp_path):
    path = tmp_path / "session.jsonl"
    recorder = RecordingProvider(_StubProvider([LLMResponse(content="ok")]), path)
    await recorder.complete(messages=_messages())

    replay = ReplayProvider(path)
    drifted = [Message(role=Role.USER, content="DIFFERENT prompt")]
    with pytest.raises(ReplayDriftError) as exc_info:
        await replay.complete(messages=drifted)
    text = str(exc_info.value)
    assert "drift" in text
    assert "---" in text and "+++" in text  # unified diff present


@pytest.mark.asyncio
async def test_replay_non_strict_returns_recorded_on_drift(tmp_path):
    path = tmp_path / "session.jsonl"
    recorder = RecordingProvider(_StubProvider([LLMResponse(content="ok")]), path)
    await recorder.complete(messages=_messages())

    replay = ReplayProvider(path, strict=False)
    response = await replay.complete(messages=[Message(role=Role.USER, content="other")])
    assert response.content == "ok"


@pytest.mark.asyncio
async def test_replay_exhaustion_both_directions(tmp_path):
    path = tmp_path / "session.jsonl"
    recorder = RecordingProvider(
        _StubProvider([LLMResponse(content="a"), LLMResponse(content="b")]), path
    )
    await recorder.complete(messages=_messages())
    await recorder.complete(messages=_messages())

    # Fewer live calls than recorded → assert_exhausted fails
    replay = ReplayProvider(path)
    await replay.complete(messages=_messages())
    with pytest.raises(ReplayDriftError, match="not consumed"):
        replay.assert_exhausted()

    # More live calls than recorded → next call fails
    await replay.complete(messages=_messages())
    with pytest.raises(ReplayDriftError, match="exhausted"):
        await replay.complete(messages=_messages())


@pytest.mark.asyncio
async def test_recording_stream_accumulates_single_exchange(tmp_path):
    path = tmp_path / "session.jsonl"
    response = LLMResponse(
        content="streamed text",
        usage=TokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10),
    )
    recorder = RecordingProvider(_StubProvider([response]), path)
    chunks = [c async for c in recorder.stream(messages=_messages())]
    assert "".join(c.content or "" for c in chunks) == "streamed text"

    exchanges = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line).get("kind") == "exchange"
    ]
    assert len(exchanges) == 1
    assert exchanges[0]["response"]["content"] == "streamed text"

    # Replaying a recorded stream yields one final chunk
    replay = ReplayProvider(path)
    replay_chunks = [c async for c in replay.stream(messages=_messages())]
    assert len(replay_chunks) == 1
    assert replay_chunks[0].done
    assert replay_chunks[0].content == "streamed text"


@pytest.mark.asyncio
async def test_recording_failure_is_fail_open(tmp_path, monkeypatch):
    """A broken recording sink must never break the LLM call."""
    path = tmp_path / "nodir" / "session.jsonl"
    recorder = RecordingProvider(_StubProvider([LLMResponse(content="ok")]), path)
    monkeypatch.setattr(
        "pathlib.Path.open", lambda *a, **k: (_ for _ in ()).throw(OSError("disk full"))
    )
    response = await recorder.complete(messages=_messages())
    assert response.content == "ok"
