"""Integration test: full agent loop with mocked LLM."""

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest

from norn.core.agent import AgentLoop
from norn.core.config import PermissionMode
from norn.core.models import LLMResponse, ToolCall
from norn.dream.engine import DreamEngine
from norn.dream.trigger import DreamTrigger
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.memory.models import MemoryConfig, SessionSummary
from norn.memory.session_logger import SessionLogger
from norn.memory.store import MemoryStore
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry


@pytest.fixture
def full_registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(FileWriteTool())
    reg.register(FileEditTool())
    reg.register(GlobTool())
    reg.register(GrepTool())
    return reg


@pytest.mark.asyncio
async def test_full_loop_write_then_read(full_registry, tmp_path):
    """LLM writes a file, then reads it back."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            # Step 1: Write file
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": str(tmp_path / "test.py"), "content": "print('hello')"},
                    )
                ],
            ),
            # Step 2: Read file back
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="file_read",
                        arguments={"path": str(tmp_path / "test.py")},
                    )
                ],
            ),
            # Step 3: Final response
            LLMResponse(content="Done! File written and verified.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=full_registry, cwd=str(tmp_path))
    result = await agent.run("write a hello world script and verify it")

    assert result.content == "Done! File written and verified."
    assert (tmp_path / "test.py").read_text() == "print('hello')"
    assert llm.complete.call_count == 3


@pytest.mark.asyncio
async def test_full_loop_bash_execution(full_registry, tmp_path):
    """LLM runs a bash command."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "echo 'norn is alive'"},
                    )
                ],
            ),
            LLMResponse(content="Command executed successfully.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=full_registry, cwd=str(tmp_path))
    result = await agent.run("run echo")
    assert result.content == "Command executed successfully."


@pytest.mark.asyncio
async def test_all_tools_registered(full_registry):
    """Verify all 6 Phase 1 tools are registered."""
    expected = {"bash", "file_read", "file_write", "file_edit", "glob", "grep"}
    actual = {t.name for t in full_registry.list_tools()}
    assert actual == expected


@pytest.mark.asyncio
async def test_schemas_valid_for_llm(full_registry):
    """Verify schemas are valid for LLM consumption."""
    schemas = full_registry.get_schemas()
    assert len(schemas) == 6
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_permission_blocks_destructive_bash():
    """Full pipeline: bash rm -rf should be denied in interactive mode."""
    checker = PermissionChecker(
        mode=PermissionMode.INTERACTIVE,
        classifier=RiskClassifier(),
    )

    registry = ToolRegistry()
    registry.register(BashTool())

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "rm -rf /"})],
            ),
            LLMResponse(content="I couldn't do that.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=registry, permission_checker=checker)
    result = await agent.run("delete everything")
    assert result.content == "I couldn't do that."


@pytest.mark.asyncio
async def test_feature_flag_hides_tool():
    """Tool behind disabled flag should not appear in schemas sent to LLM."""
    flag_registry = FeatureFlagRegistry(flags={"ml_tools": FeatureFlag("ml_tools", False)})
    registry = ToolRegistry(flag_registry=flag_registry)

    registry.register(BashTool(), feature_flag="ml_tools")

    schemas = registry.get_schemas()
    assert len(schemas) == 0  # Tool hidden

    # Enable the flag
    flag_registry.apply_config({"ml_tools": True})
    registry._schema_cache = None  # Force rebuild
    schemas = registry.get_schemas()
    assert len(schemas) == 1


@pytest.mark.asyncio
async def test_yolo_mode_allows_destructive():
    """Yolo mode should allow everything, even destructive commands."""
    checker = PermissionChecker(
        mode=PermissionMode.YOLO,
        classifier=RiskClassifier(),
    )

    registry = ToolRegistry()
    registry.register(BashTool())

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="bash", arguments={"command": "echo safe"})],
            ),
            LLMResponse(content="Done.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=registry, permission_checker=checker)
    result = await agent.run("run a command")
    assert result.content == "Done."


@pytest.mark.asyncio
async def test_agent_with_memory_injects_context(tmp_path):
    """Full pipeline: memory content appears in system prompt sent to LLM."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.write_memory("# Memory\n\n- User's project is called Norn\n")

    registry = ToolRegistry()
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value=LLMResponse(content="Your project is Norn!", tool_calls=[])
    )

    agent = AgentLoop(llm=llm, registry=registry, memory_store=store)
    await agent.run("What's my project?")

    call_args = llm.complete.call_args
    messages = call_args.kwargs.get("messages") or call_args.args[0]
    system_content = messages[0].content
    assert "Norn" in system_content


@pytest.mark.asyncio
async def test_dream_engine_full_cycle(tmp_path):
    """Full dream cycle: orient -> gather -> consolidate -> prune."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    store.append_daily("2026-03-31", "Session 1: built the dream system")

    response_json = {
        "memory": "# Norn Memory\n\n- Built dream system on 2026-03-31\n",
        "topics": {},
        "pruned_topics": [],
        "summary": "Recorded dream system work.",
    }
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content=json.dumps(response_json)))

    engine = DreamEngine(store=store, llm=llm)
    result = await engine.dream()
    assert result.success is True
    assert "dream system" in store.read_memory().lower()


def test_dream_trigger_gates(tmp_path):
    """Integration: trigger respects all three gates."""
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir()

    trigger = DreamTrigger(
        memory_dir=memory_dir,
        interval_hours=24,
        min_sessions=5,
    )

    # Should dream: no previous dream + enough sessions
    assert trigger.should_dream(session_count=5) is True
    trigger.release_lock()

    # Record dream time as now
    trigger.write_last_dream_time(datetime.now(tz=UTC))

    # Should NOT dream: too recent
    assert trigger.should_dream(session_count=10) is False


@pytest.mark.asyncio
async def test_session_logger_records_and_counts(tmp_path):
    """Session logger records summaries and tracks count."""
    config = MemoryConfig(memory_dir=tmp_path / "memory")
    store = MemoryStore(config)
    store.ensure_dirs()
    logger = SessionLogger(store)

    for i in range(3):
        summary = SessionSummary(
            session_id=f"s{i}",
            started_at=datetime(2026, 3, 31, 10 + i, 0, tzinfo=UTC),
            ended_at=datetime(2026, 3, 31, 10 + i, 30, tzinfo=UTC),
            user_messages=3,
            tool_calls=5,
            summary=f"Session {i} summary.",
        )
        logger.log_session(summary)

    assert logger.session_count() == 3
    daily_content = store.read_daily("2026-03-31")
    assert "s0" in daily_content
    assert "s1" in daily_content
    assert "s2" in daily_content
