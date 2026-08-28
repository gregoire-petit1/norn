"""End-to-end replay: a recorded session drives a full AgentLoop run.

The test records a two-exchange session (tool call, then final answer)
against a stub provider, then replays it through a *fresh* AgentLoop in
strict mode — proving that the loop's outbound requests are deterministic
and that replay needs no API key. Agent construction discipline matters:
``env_bootstrap=False, repo_map=False, memory_store=None`` keeps the system
prompt byte-identical across runs (machine-dependent context would drift
the request hash — that is the point of strict replay).
"""

import pytest
from pydantic import BaseModel

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse, StreamChunk, ToolCall
from norn.core.replay import RecordingProvider, ReplayDriftError, ReplayProvider
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class EchoInput(BaseModel):
    text: str


class EchoTool:
    name = "echo"
    description = "Echo text back"
    risk_level = RiskLevel.LOW
    input_model = EchoInput

    async def execute(self, input: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"echo: {input.text}")


class _ScriptedProvider:
    """Yields scripted responses; only used for the recording pass."""

    model = "scripted/model"

    def __init__(self) -> None:
        self._responses = [
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="1", name="echo", arguments={"text": "ping"})],
            ),
            LLMResponse(content="pong received"),
        ]

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return self._responses.pop(0)

    async def stream(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        response = await self.complete(messages, tools, temperature, max_tokens)
        yield StreamChunk(
            content=response.content,
            tool_calls=response.tool_calls or None,
            done=True,
        )


def _make_agent(provider) -> AgentLoop:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return AgentLoop(
        llm=provider,
        registry=reg,
        env_bootstrap=False,
        repo_map=False,
        memory_store=None,
    )


@pytest.mark.asyncio
async def test_recorded_session_replays_full_agent_loop(tmp_path):
    path = tmp_path / "e2e.jsonl"

    # Pass 1: record a live-shaped run
    recording_agent = _make_agent(RecordingProvider(_ScriptedProvider(), path))
    recorded = await recording_agent.run("say ping")
    assert recorded.content == "pong received"

    # Pass 2: a fresh agent replays the session with zero API access
    replay = ReplayProvider(path, strict=True)
    replay_agent = _make_agent(replay)
    replayed = await replay_agent.run("say ping")

    assert replayed.content == "pong received"
    replay.assert_exhausted()
    # The replayed thread went through the tool round for real
    tool_msgs = [m for m in replay_agent.history if m.tool_call_id]
    assert tool_msgs and tool_msgs[0].content == "echo: ping"


@pytest.mark.asyncio
async def test_replay_detects_agent_behaviour_drift(tmp_path):
    path = tmp_path / "e2e.jsonl"
    recording_agent = _make_agent(RecordingProvider(_ScriptedProvider(), path))
    await recording_agent.run("say ping")

    # Same recording, but the agent is constructed differently → its first
    # request no longer matches the recorded one: strict replay must fail.
    replay = ReplayProvider(path, strict=True)
    drifted_agent = _make_agent(replay)
    drifted_agent.system_prompt = "A COMPLETELY DIFFERENT SYSTEM PROMPT"
    with pytest.raises(ReplayDriftError, match="drift"):
        await drifted_agent.run("say ping")
