"""Tests for the byte-stable system prompt layout (SOTA v2, workstream D)."""

from unittest.mock import AsyncMock

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse
from norn.core.prompts import CACHE_BREAK
from norn.tools.registry import ToolRegistry


class _FakeMemoryStore:
    """Duck-typed MemoryStore: only read_memory/read_lessons are used."""

    def __init__(self, memory: str = "", lessons: str = "") -> None:
        self.memory = memory
        self.lessons = lessons

    def read_memory(self) -> str:
        return self.memory

    def read_lessons(self) -> str:
        return self.lessons


def _make_agent(*, stable_prompt: bool, store: _FakeMemoryStore | None = None) -> AgentLoop:
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(content="ok"))
    return AgentLoop(
        llm=llm,
        registry=ToolRegistry(),
        system_prompt="BASE PROMPT",
        memory_store=store,
        env_bootstrap=False,
        repo_map=False,
        stable_prompt=stable_prompt,
    )


def test_default_layout_unchanged_without_flag():
    store = _FakeMemoryStore(memory="fact A", lessons="lesson B")
    agent = _make_agent(stable_prompt=False, store=store)
    prompt = agent._build_system_prompt()
    assert CACHE_BREAK not in prompt
    assert prompt.startswith("BASE PROMPT")
    assert "## Persistent Memory" in prompt
    assert "## Learned Lessons" in prompt


def test_stable_layout_puts_volatile_behind_cache_break():
    store = _FakeMemoryStore(memory="fact A", lessons="lesson B")
    agent = _make_agent(stable_prompt=True, store=store)
    prompt = agent._build_system_prompt()
    assert CACHE_BREAK in prompt
    static, _, volatile = prompt.partition(CACHE_BREAK)
    assert static == "BASE PROMPT"
    assert "fact A" in volatile
    assert "lesson B" in volatile


def test_stable_layout_static_prefix_byte_equal_across_memory_change():
    store = _FakeMemoryStore(memory="fact A")
    agent = _make_agent(stable_prompt=True, store=store)
    static_1 = agent._build_system_prompt().partition(CACHE_BREAK)[0]

    store.memory = "fact A plus something new"
    static_2 = agent._build_system_prompt().partition(CACHE_BREAK)[0]
    assert static_1 == static_2


def test_stable_layout_no_break_when_no_volatile_content():
    agent = _make_agent(stable_prompt=True, store=None)
    prompt = agent._build_system_prompt()
    assert CACHE_BREAK not in prompt
    assert prompt == "BASE PROMPT"


def test_stable_layout_env_and_repo_map_in_static_prefix():
    store = _FakeMemoryStore(memory="fact A")
    agent = _make_agent(stable_prompt=True, store=store)
    # Simulate session-static injections (normally set in __init__)
    agent._env_snapshot = "[Environment]\nos: test"
    agent._repo_map = "[Repo Map]\nsrc/x.py"
    agent._system_prompt_cache = None  # invalidate W1.1 cache
    agent._last_prompt_key = ("", "")

    prompt = agent._build_system_prompt()
    static, _, volatile = prompt.partition(CACHE_BREAK)
    assert "[Environment]" in static
    assert "[Repo Map]" in static
    assert "fact A" in volatile
