"""Record/replay of LLM exchanges (SOTA v2, workstream A — dsh-style).

A recorded session is a JSONL file: one ``meta`` line, then one ``exchange``
line per LLM call. The file serves both as replay *input* (the responses to
feed back) and as the *oracle* (the request hash at each step) — any drift in
what the agent sends the model surfaces as a unified diff, not a silent pass.

``ReplayProvider`` never touches litellm: replay-driven tests run with zero
API keys. Streaming fidelity note: replay emits a single final chunk (content
+ tool_calls + usage), not the original chunk boundaries — sufficient to
exercise the agent loop, by design.

Secrets note: recordings capture full prompts (which may embed file contents);
the observability redaction pipeline does NOT apply here. Review a recording
before committing it as a test fixture.
"""

from __future__ import annotations

import contextlib
import difflib
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from norn.core.models import LLMResponse, Message, StreamChunk, TokenUsage, ToolCall
from norn.observability import EventName, get_logger

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from norn.core.llm import LLMProvider

_log = get_logger(__name__)

REPLAY_FORMAT_VERSION = 1


class ReplayDriftError(AssertionError):
    """The live request diverged from the recorded one (carries a diff)."""


def canonical_request(
    messages: list[Message],
    tools: list[dict] | None,
    temperature: float,
    max_tokens: int,
) -> str:
    """Canonical JSON of a request: stable key order, compact separators."""
    payload = {
        "messages": [m.model_dump(exclude_none=True) for m in messages],
        "tools": tools or [],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def request_hash(canonical: str) -> str:
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _response_to_dict(response: LLMResponse) -> dict[str, Any]:
    # latency_ms deliberately excluded: replays are deterministic.
    return {
        "content": response.content,
        "tool_calls": [tc.model_dump() for tc in response.tool_calls],
        "usage": response.usage.model_dump() if response.usage else None,
        "model": response.model,
    }


def _response_from_dict(data: dict[str, Any]) -> LLMResponse:
    return LLMResponse(
        content=data.get("content"),
        tool_calls=[ToolCall(**tc) for tc in data.get("tool_calls") or []],
        usage=TokenUsage(**data["usage"]) if data.get("usage") else None,
        model=data.get("model"),
    )


class RecordingProvider:
    """LLMProvider wrapper that appends every exchange to a JSONL file.

    Delegates to ``inner``; recording failures are fail-open (a full disk
    must never break a turn). Streamed calls are accumulated and recorded
    as a single assembled exchange.
    """

    def __init__(self, inner: LLMProvider, path: Path) -> None:
        self._inner = inner
        self._path = Path(path)
        self._seq = 0
        self._meta_written = False

    def _write_line(self, record: dict[str, Any]) -> None:
        with contextlib.suppress(Exception):
            self._path.parent.mkdir(parents=True, exist_ok=True)
            if not self._meta_written:
                meta = {
                    "kind": "meta",
                    "version": REPLAY_FORMAT_VERSION,
                    "model": getattr(self._inner, "model", None),
                }
                with self._path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(meta, ensure_ascii=False) + "\n")
                self._meta_written = True
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

    def _record(
        self,
        messages: list[Message],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
        response: LLMResponse,
    ) -> None:
        canonical = canonical_request(messages, tools, temperature, max_tokens)
        rec = {
            "kind": "exchange",
            "seq": self._seq,
            "request_hash": request_hash(canonical),
            "request": json.loads(canonical),
            "response": _response_to_dict(response),
        }
        self._write_line(rec)
        with contextlib.suppress(Exception):
            _log.info(EventName.LLM_EXCHANGE, seq=self._seq, request_hash=rec["request_hash"])
        self._seq += 1

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        response = await self._inner.complete(
            messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        self._record(messages, tools, temperature, max_tokens, response)
        return response

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        content_parts: list[str] = []
        tool_calls: list[ToolCall] = []
        usage: TokenUsage | None = None
        async for chunk in self._inner.stream(
            messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        ):
            if chunk.content:
                content_parts.append(chunk.content)
            if chunk.done:
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls
                usage = chunk.usage
            yield chunk
        assembled = LLMResponse(
            content="".join(content_parts) or None,
            tool_calls=tool_calls,
            usage=usage,
            model=getattr(self._inner, "model", None),
        )
        self._record(messages, tools, temperature, max_tokens, assembled)


class ReplayProvider:
    """Deterministic LLMProvider that replays a recorded session.

    Exchanges are consumed sequentially. In strict mode (default) a request
    whose hash differs from the recording raises :class:`ReplayDriftError`
    with a unified diff of the canonical request JSON; with ``strict=False``
    the mismatch is logged as a warning and the recorded response is
    returned anyway (pure sequential matching).
    """

    def __init__(self, path: Path, *, strict: bool = True) -> None:
        self._path = Path(path)
        self._strict = strict
        self._exchanges: list[dict[str, Any]] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("kind") == "exchange":
                self._exchanges.append(record)
        self._cursor = 0

    def _next_exchange(
        self,
        messages: list[Message],
        tools: list[dict] | None,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        if self._cursor >= len(self._exchanges):
            raise ReplayDriftError(
                f"replay exhausted: live run made more LLM calls than the "
                f"{len(self._exchanges)} recorded in {self._path.name}"
            )
        recorded = self._exchanges[self._cursor]
        self._cursor += 1

        canonical = canonical_request(messages, tools, temperature, max_tokens)
        live_hash = request_hash(canonical)
        if live_hash != recorded["request_hash"]:
            expected = json.dumps(recorded["request"], sort_keys=True, indent=2, ensure_ascii=False)
            actual = json.dumps(json.loads(canonical), sort_keys=True, indent=2, ensure_ascii=False)
            diff = "\n".join(
                difflib.unified_diff(
                    expected.splitlines(),
                    actual.splitlines(),
                    fromfile=f"recorded (seq={recorded['seq']})",
                    tofile="live",
                    lineterm="",
                )
            )
            if self._strict:
                raise ReplayDriftError(
                    f"request drift at exchange seq={recorded['seq']}:\n{diff}"
                )
            with contextlib.suppress(Exception):
                _log.warning(
                    "replay_request_drift", seq=recorded["seq"], live_hash=live_hash
                )
        return recorded

    def assert_exhausted(self) -> None:
        """Fail if recorded exchanges were not all consumed (flow drift)."""
        remaining = len(self._exchanges) - self._cursor
        if remaining:
            raise ReplayDriftError(
                f"{remaining} recorded exchange(s) not consumed — the live run "
                f"made fewer LLM calls than the recording in {self._path.name}"
            )

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        recorded = self._next_exchange(messages, tools, temperature, max_tokens)
        return _response_from_dict(recorded["response"])

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        response = await self.complete(
            messages=messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )
        yield StreamChunk(
            content=response.content,
            tool_calls=response.tool_calls or None,
            done=True,
            usage=response.usage,
        )
