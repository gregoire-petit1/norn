"""Tests for the dream engine."""

import json
from unittest.mock import AsyncMock

import pytest

from norn.core.models import LLMResponse
from norn.dream.engine import DreamEngine, DreamResult
from norn.memory.models import MemoryConfig
from norn.memory.store import MemoryStore


@pytest.fixture
def memory_dir(tmp_path):
    return tmp_path / "memory"


@pytest.fixture
def store(memory_dir):
    config = MemoryConfig(memory_dir=memory_dir)
    s = MemoryStore(config)
    s.ensure_dirs()
    return s


@pytest.fixture
def mock_llm():
    """Mock LLM that returns valid consolidation JSON."""
    llm = AsyncMock()
    response_json = {
        "memory": "# Norn Memory\n\n- User works on project Norn\n- Prefers Python + uv\n",
        "topics": {"norn": "# Norn Project\n\nA coding agent built from scratch.\n"},
        "pruned_topics": [],
        "summary": "Added Norn project info from recent sessions.",
    }
    llm.complete = AsyncMock(return_value=LLMResponse(content=json.dumps(response_json)))
    return llm


@pytest.fixture
def engine(store, mock_llm):
    return DreamEngine(store=store, llm=mock_llm)


class TestOrientPhase:
    def test_reads_current_memory(self, engine, store):
        """Orient phase reads MEMORY.md."""
        context = engine._orient()
        assert "Norn Memory" in context["current_memory"]

    def test_reads_topic_files(self, engine, store):
        """Orient phase reads all topic files."""
        store.write_topic("auth", "# Auth\n\nJWT tokens.\n")
        context = engine._orient()
        assert "auth" in context["topic_files"]
        assert "JWT" in context["topic_files"]["auth"]

    def test_reads_empty_state(self, engine):
        """Orient phase handles empty memory."""
        context = engine._orient()
        assert context["topic_files"] == {}


class TestGatherPhase:
    def test_gathers_daily_logs(self, engine, store):
        """Gather phase collects daily logs."""
        store.append_daily("2026-03-31", "Session 1: did X")
        store.append_daily("2026-03-30", "Session 2: did Y")
        logs = engine._gather()
        assert "2026-03-31" in logs
        assert "2026-03-30" in logs

    def test_gathers_empty_logs(self, engine):
        """Gather phase handles no logs."""
        logs = engine._gather()
        assert logs == {}


class TestConsolidatePhase:
    @pytest.mark.asyncio
    async def test_calls_llm(self, engine, mock_llm, store):
        """Consolidate phase calls the LLM with context."""
        store.append_daily("2026-03-31", "Session 1: built agent")
        result = await engine._consolidate(
            current_memory="# Norn Memory\n",
            daily_logs={"2026-03-31": "Session 1: built agent"},
            topic_files={},
        )
        mock_llm.complete.assert_called_once()
        assert result is not None
        assert "memory" in result

    @pytest.mark.asyncio
    async def test_parses_json_response(self, engine, store):
        """Consolidate phase parses the LLM JSON response."""
        result = await engine._consolidate(
            current_memory="",
            daily_logs={},
            topic_files={},
        )
        assert "Norn Memory" in result["memory"]
        assert "norn" in result["topics"]


class TestPrunePhase:
    def test_applies_memory_update(self, engine, store):
        """Prune phase writes updated MEMORY.md."""
        engine._prune(
            consolidation={
                "memory": "# Updated Memory\n\n- new fact\n",
                "topics": {},
                "pruned_topics": [],
                "summary": "test",
            }
        )
        content = store.read_memory()
        assert "Updated Memory" in content
        assert "new fact" in content

    def test_applies_topic_updates(self, engine, store):
        """Prune phase writes updated topic files."""
        engine._prune(
            consolidation={
                "memory": "# Memory\n",
                "topics": {"auth": "# Auth\n\nOAuth2 now.\n"},
                "pruned_topics": [],
                "summary": "test",
            }
        )
        content = store.read_topic("auth")
        assert "OAuth2" in content

    def test_prunes_stale_topics(self, engine, store):
        """Prune phase deletes stale topic files."""
        store.write_topic("old-project", "# Old\n\nNo longer relevant.\n")
        assert store.read_topic("old-project") is not None

        engine._prune(
            consolidation={
                "memory": "# Memory\n",
                "topics": {},
                "pruned_topics": ["old-project"],
                "summary": "test",
            }
        )
        assert store.read_topic("old-project") is None


class TestFullDream:
    @pytest.mark.asyncio
    async def test_dream_end_to_end(self, engine, store):
        """Full dream cycle: orient -> gather -> consolidate -> prune."""
        store.append_daily("2026-03-31", "Session 1: worked on agent")
        result = await engine.dream()
        assert isinstance(result, DreamResult)
        assert result.success is True
        assert result.summary is not None
        # Memory should be updated
        content = store.read_memory()
        assert "Norn" in content

    @pytest.mark.asyncio
    async def test_dream_handles_llm_error(self, store):
        """Dream handles LLM errors gracefully."""
        bad_llm = AsyncMock()
        bad_llm.complete = AsyncMock(side_effect=Exception("API error"))
        engine = DreamEngine(store=store, llm=bad_llm)
        result = await engine.dream()
        assert result.success is False
        assert "API error" in result.error

    @pytest.mark.asyncio
    async def test_dream_handles_invalid_json(self, store):
        """Dream handles invalid JSON from LLM."""
        bad_llm = AsyncMock()
        bad_llm.complete = AsyncMock(return_value=LLMResponse(content="this is not json"))
        engine = DreamEngine(store=store, llm=bad_llm)
        result = await engine.dream()
        assert result.success is False
        assert result.error is not None
