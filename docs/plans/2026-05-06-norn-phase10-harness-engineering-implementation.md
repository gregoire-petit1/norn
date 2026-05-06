# Phase 10 — Harness Engineering Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement 4 harness-level improvements (env bootstrap, per-turn output budget, sliding window + summarization, domain-aware routing) to reduce wasted LLM calls and control context growth.

**Architecture:** Purely additive modifications to the agent loop. Each feature is a new module wired into `AgentLoop` via optional composition. Config-gated with sensible defaults. Feature 3 (sliding window) is opt-in initially.

**Tech Stack:** Python 3.11+, Pydantic, asyncio, structlog, litellm (token counting), pytest + pytest-asyncio.

**Design doc:** `docs/plans/2026-05-06-norn-phase10-harness-engineering-design.md`

---

## Task 1: Env Bootstrap — Core Module

**Files:**
- Create: `src/norn/core/env_bootstrap.py`
- Test: `tests/test_core/test_env_bootstrap.py`

**Step 1: Write the failing tests**

```python
# tests/test_core/test_env_bootstrap.py
"""Tests for environment bootstrap module."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from norn.core.env_bootstrap import EnvironmentSnapshot, scan_environment


class TestEnvironmentSnapshot:
    """Test EnvironmentSnapshot rendering."""

    def test_render_minimal(self):
        """Snapshot with minimal info renders without crashing."""
        snap = EnvironmentSnapshot(
            cwd="/tmp/test",
            git_branch=None,
            git_dirty=False,
            git_recent_commits=[],
            languages=[],
            package_managers=[],
            key_files=[],
            directory_tree="",
            python_version=None,
            venv_active=False,
        )
        rendered = snap.render()
        assert "[Environment]" in rendered
        assert "/tmp/test" in rendered

    def test_render_full(self):
        """Snapshot with all info renders all sections."""
        snap = EnvironmentSnapshot(
            cwd="/home/user/project",
            git_branch="feature/env-boot",
            git_dirty=True,
            git_recent_commits=["abc1234 feat: add env boot", "def5678 fix: typo"],
            languages=["python", "typescript"],
            package_managers=["uv"],
            key_files=["pyproject.toml", "package.json"],
            directory_tree="src/ tests/ docs/",
            python_version="3.11.9",
            venv_active=True,
        )
        rendered = snap.render()
        assert "feature/env-boot" in rendered
        assert "dirty" in rendered.lower()
        assert "python" in rendered
        assert "uv" in rendered

    def test_render_char_limit(self):
        """Rendered snapshot should not exceed 600 chars."""
        snap = EnvironmentSnapshot(
            cwd="/very/long/path/that/is/deep",
            git_branch="main",
            git_dirty=False,
            git_recent_commits=["a" * 100, "b" * 100, "c" * 100],
            languages=["python", "typescript", "rust", "go"],
            package_managers=["uv", "npm", "cargo"],
            key_files=["pyproject.toml", "package.json", "Cargo.toml", "go.mod"],
            directory_tree="src/ tests/ docs/ lib/ pkg/ cmd/ internal/",
            python_version="3.11.9",
            venv_active=True,
        )
        rendered = snap.render()
        assert len(rendered) <= 600


class TestScanEnvironment:
    """Test scan_environment function."""

    def test_scan_returns_snapshot(self, tmp_path: Path):
        """scan_environment returns an EnvironmentSnapshot."""
        result = scan_environment(str(tmp_path))
        assert isinstance(result, EnvironmentSnapshot)
        assert result.cwd == str(tmp_path)

    def test_scan_detects_python_project(self, tmp_path: Path):
        """Detects Python when pyproject.toml exists."""
        (tmp_path / "pyproject.toml").write_text("[project]\nname='test'\n")
        (tmp_path / "main.py").write_text("print('hi')")
        result = scan_environment(str(tmp_path))
        assert "python" in result.languages
        assert "pyproject.toml" in result.key_files

    def test_scan_detects_git(self, tmp_path: Path):
        """Detects git branch when .git exists."""
        (tmp_path / ".git").mkdir()
        # Mock subprocess to avoid needing real git
        with patch("norn.core.env_bootstrap._run_git") as mock_git:
            mock_git.side_effect = lambda cmd, cwd: {
                ["rev-parse", "--abbrev-ref", "HEAD"]: "main",
                ["status", "--porcelain"]: "",
                ["log", "--oneline", "-3"]: "abc feat: init\ndef fix: typo",
            }.get(cmd, "")
            result = scan_environment(str(tmp_path))
        assert result.git_branch == "main"
        assert result.git_dirty is False

    def test_scan_graceful_on_failure(self, tmp_path: Path):
        """scan_environment never raises, even on broken state."""
        # A directory with no git, no files → still returns valid snapshot
        result = scan_environment(str(tmp_path))
        assert result.git_branch is None
        assert result.languages == []

    def test_scan_detects_venv(self, tmp_path: Path):
        """Detects virtual environment."""
        (tmp_path / ".venv").mkdir()
        result = scan_environment(str(tmp_path))
        assert result.venv_active is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_env_bootstrap.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'norn.core.env_bootstrap'`

**Step 3: Write the implementation**

```python
# src/norn/core/env_bootstrap.py
"""Environment bootstrap — scan project context before first LLM call.

Provides a structured snapshot of the working directory (git state,
languages, package managers, directory layout) to inject into the
system prompt, eliminating 2-4 exploratory tool calls per session.
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Language detection: extension → language name
_LANG_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".js": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".cpp": "c++",
    ".c": "c",
}

# Key config files to detect
_KEY_FILES = [
    "pyproject.toml",
    "setup.py",
    "requirements.txt",
    "package.json",
    "tsconfig.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "Dockerfile",
    "docker-compose.yml",
    ".env",
]

# Package manager detection: file → manager
_PACKAGE_MANAGERS: dict[str, str] = {
    "pyproject.toml": "uv/pip",
    "requirements.txt": "pip",
    "package.json": "npm",
    "pnpm-lock.yaml": "pnpm",
    "yarn.lock": "yarn",
    "Cargo.toml": "cargo",
    "go.mod": "go",
}

_MAX_RENDER_CHARS = 600
_GIT_TIMEOUT = 2  # seconds


@dataclass
class EnvironmentSnapshot:
    """Structured project environment information."""

    cwd: str
    git_branch: str | None = None
    git_dirty: bool = False
    git_recent_commits: list[str] = field(default_factory=list)
    languages: list[str] = field(default_factory=list)
    package_managers: list[str] = field(default_factory=list)
    key_files: list[str] = field(default_factory=list)
    directory_tree: str = ""
    python_version: str | None = None
    venv_active: bool = False

    def render(self) -> str:
        """Render as a compact text block for system prompt injection."""
        lines: list[str] = [f"[Environment] cwd: {self.cwd}"]

        if self.git_branch:
            state = "dirty" if self.git_dirty else "clean"
            lines.append(f"Git: {self.git_branch} ({state})")
            if self.git_recent_commits:
                # Only first 50 chars per commit
                commits = [c[:50] for c in self.git_recent_commits[:3]]
                lines.append(f"Recent: {' | '.join(commits)}")

        if self.languages:
            lines.append(f"Languages: {', '.join(self.languages)}")

        if self.package_managers:
            lines.append(f"Pkg: {', '.join(self.package_managers)}")

        if self.key_files:
            lines.append(f"Files: {', '.join(self.key_files)}")

        if self.directory_tree:
            lines.append(f"Layout: {self.directory_tree}")

        if self.python_version:
            venv_marker = " (venv)" if self.venv_active else ""
            lines.append(f"Python: {self.python_version}{venv_marker}")
        elif self.venv_active:
            lines.append("Venv: active")

        rendered = "\n".join(lines)
        # Hard cap to prevent bloat
        if len(rendered) > _MAX_RENDER_CHARS:
            rendered = rendered[:_MAX_RENDER_CHARS - 3] + "..."
        return rendered


def _run_git(cmd: list[str], cwd: str) -> str:
    """Run a git command, returning stdout or empty string on failure."""
    try:
        result = subprocess.run(
            ["git", *cmd],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
            cwd=cwd,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass
    return ""


def scan_environment(cwd: str) -> EnvironmentSnapshot:
    """Scan the working directory and return an EnvironmentSnapshot.

    Never raises — all detection is best-effort with graceful fallback.
    Designed to complete in <200ms.
    """
    path = Path(cwd)
    snap = EnvironmentSnapshot(cwd=cwd)

    # --- Git detection ---
    if (path / ".git").exists():
        snap.git_branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], cwd) or None
        porcelain = _run_git(["status", "--porcelain"], cwd)
        snap.git_dirty = bool(porcelain)
        log_output = _run_git(["log", "--oneline", "-3"], cwd)
        if log_output:
            snap.git_recent_commits = log_output.splitlines()[:3]

    # --- File scanning ---
    try:
        entries = os.listdir(cwd)
    except OSError:
        entries = []

    # Key files
    snap.key_files = [f for f in _KEY_FILES if f in entries]

    # Package managers
    detected_managers: list[str] = []
    for filename, manager in _PACKAGE_MANAGERS.items():
        if filename in entries:
            detected_managers.append(manager)
    snap.package_managers = sorted(set(detected_managers))

    # Language detection (scan top-level + src/ if exists)
    detected_langs: set[str] = set()
    scan_dirs = [path]
    src_dir = path / "src"
    if src_dir.is_dir():
        scan_dirs.append(src_dir)

    for scan_path in scan_dirs:
        try:
            for entry in os.scandir(scan_path):
                if entry.is_file():
                    ext = Path(entry.name).suffix.lower()
                    if ext in _LANG_EXTENSIONS:
                        detected_langs.add(_LANG_EXTENSIONS[ext])
        except OSError:
            continue
    snap.languages = sorted(detected_langs)

    # Directory tree (top-level dirs only)
    try:
        dirs = sorted(
            e for e in entries
            if (path / e).is_dir() and not e.startswith(".")
        )[:10]
        snap.directory_tree = " ".join(f"{d}/" for d in dirs)
    except OSError:
        pass

    # Python version
    if "python" in snap.languages or any(
        f in snap.key_files for f in ("pyproject.toml", "requirements.txt", "setup.py")
    ):
        snap.python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    # Venv detection
    snap.venv_active = bool(os.environ.get("VIRTUAL_ENV")) or (path / ".venv").is_dir()

    return snap
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_env_bootstrap.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/env_bootstrap.py tests/test_core/test_env_bootstrap.py
git commit -m "feat(env-bootstrap): add environment scanning module

Scans cwd for git state, languages, package managers, and directory
layout. Renders a compact snapshot (<600 chars) for system prompt
injection. All detection is best-effort with graceful fallback."
```

---

## Task 2: Env Bootstrap — Wire into AgentLoop

**Files:**
- Modify: `src/norn/core/agent.py:44-81` (constructor + `_build_system_prompt`)
- Modify: `src/norn/core/config.py:104-108` (add `env_bootstrap` field)
- Modify: `configs/default.yaml`
- Test: `tests/test_core/test_env_bootstrap.py` (add integration test)

**Step 1: Write the failing test**

```python
# Add to tests/test_core/test_env_bootstrap.py

class TestAgentIntegration:
    """Test env bootstrap integration with AgentLoop."""

    def test_system_prompt_includes_snapshot(self, tmp_path: Path):
        """When env_bootstrap=True, system prompt contains [Environment] block."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop

        mock_llm = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd=str(tmp_path),
            env_bootstrap=True,
        )
        prompt = loop._build_system_prompt()
        assert "[Environment]" in prompt

    def test_system_prompt_excludes_snapshot_when_disabled(self, tmp_path: Path):
        """When env_bootstrap=False, no [Environment] block."""
        from unittest.mock import MagicMock

        from norn.core.agent import AgentLoop

        mock_llm = MagicMock()
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd=str(tmp_path),
            env_bootstrap=False,
        )
        prompt = loop._build_system_prompt()
        assert "[Environment]" not in prompt
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_env_bootstrap.py::TestAgentIntegration -v`
Expected: FAIL with `TypeError: AgentLoop.__init__() got an unexpected keyword argument 'env_bootstrap'`

**Step 3: Modify AgentLoop and AgentConfig**

In `src/norn/core/config.py`, add to `AgentConfig`:
```python
class AgentConfig(BaseModel):
    max_tool_rounds: int = 25
    minify_tool_schemas: bool = True
    max_tool_result_chars: int = 8000
    env_bootstrap: bool = True  # Phase 10: inject environment snapshot
```

In `src/norn/core/agent.py`:
- Add `env_bootstrap: bool = True` parameter to `__init__`
- In `_build_system_prompt()`, append snapshot if enabled:

```python
from norn.core.env_bootstrap import scan_environment

def __init__(self, ..., env_bootstrap: bool = True):
    ...
    self._env_bootstrap = env_bootstrap
    self._env_snapshot: str | None = None
    if env_bootstrap:
        snapshot = scan_environment(cwd)
        self._env_snapshot = snapshot.render()

def _build_system_prompt(self) -> str:
    prompt = self.system_prompt
    if self.memory_store is not None:
        memory_content = self.memory_store.read_memory()
        if memory_content.strip():
            prompt += "\n\n## Persistent Memory\n\n" + memory_content
    if self._env_snapshot:
        prompt += "\n\n" + self._env_snapshot
    return prompt
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_env_bootstrap.py -v`
Expected: All tests PASS

**Step 5: Run full test suite to check no regressions**

Run: `uv run pytest --tb=short -q`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add src/norn/core/agent.py src/norn/core/config.py tests/test_core/test_env_bootstrap.py configs/default.yaml
git commit -m "feat(env-bootstrap): wire into AgentLoop system prompt

AgentLoop now accepts env_bootstrap=True (default). On init, scans cwd
and appends a compact [Environment] block to the system prompt.
Eliminates 2-4 exploratory tool calls per session."
```

---

## Task 3: Per-Turn Output Budget — Core Module

**Files:**
- Create: `src/norn/core/turn_budget.py`
- Test: `tests/test_core/test_turn_budget.py`

**Step 1: Write the failing tests**

```python
# tests/test_core/test_turn_budget.py
"""Tests for per-turn output budget tracker."""

from __future__ import annotations

import pytest

from norn.core.turn_budget import TurnBudgetTracker


class TestTurnBudgetTracker:
    """Test TurnBudgetTracker budget allocation."""

    def test_first_call_gets_full_budget(self):
        """First tool call gets min(per_tool, per_turn) budget."""
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        text = "x" * 5000
        result = tracker.allocate(text, max_per_tool=8000)
        assert result == text  # under both limits

    def test_large_output_truncated_by_per_tool(self):
        """Output exceeding per-tool limit is truncated."""
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        text = "x" * 10000
        result = tracker.allocate(text, max_per_tool=8000)
        assert len(result) <= 8000

    def test_budget_depletes_across_calls(self):
        """Budget decreases with each allocation."""
        tracker = TurnBudgetTracker(max_chars_per_turn=10000)
        # First call: 6000 chars
        text1 = "a" * 6000
        r1 = tracker.allocate(text1, max_per_tool=8000)
        assert r1 == text1
        # Second call: 6000 chars but only 4000 remaining
        text2 = "b" * 6000
        r2 = tracker.allocate(text2, max_per_tool=8000)
        assert len(r2) <= 4000

    def test_budget_exhausted_returns_marker(self):
        """When budget is fully exhausted, returns marker message."""
        tracker = TurnBudgetTracker(max_chars_per_turn=100)
        # Exhaust budget
        tracker.allocate("x" * 100, max_per_tool=200)
        # Next call should get marker
        result = tracker.allocate("y" * 500, max_per_tool=8000)
        assert "budget exhausted" in result.lower()

    def test_reset_restores_budget(self):
        """reset() restores full budget for next turn."""
        tracker = TurnBudgetTracker(max_chars_per_turn=100)
        tracker.allocate("x" * 100, max_per_tool=200)
        tracker.reset()
        # Now full budget available again
        text = "y" * 50
        result = tracker.allocate(text, max_per_tool=8000)
        assert result == text

    def test_remaining_property(self):
        """remaining reports chars left in budget."""
        tracker = TurnBudgetTracker(max_chars_per_turn=10000)
        assert tracker.remaining == 10000
        tracker.allocate("x" * 3000, max_per_tool=8000)
        assert tracker.remaining == 7000
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_turn_budget.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'norn.core.turn_budget'`

**Step 3: Write the implementation**

```python
# src/norn/core/turn_budget.py
"""Per-turn output budget tracker (Phase 10 Feature 2).

Caps total chars of tool output injected per LLM round. When multiple
tool calls would exceed the budget, applies progressive truncation.
"""

from __future__ import annotations

from norn.core.truncation import truncate_tool_output


class TurnBudgetTracker:
    """Track and enforce per-turn tool output budget.

    Usage:
        tracker = TurnBudgetTracker(max_chars_per_turn=30000)
        # At start of each LLM round:
        tracker.reset()
        # For each tool result:
        content = tracker.allocate(raw_output, max_per_tool=8000)
    """

    def __init__(self, max_chars_per_turn: int = 30_000) -> None:
        self.max_chars_per_turn = max_chars_per_turn
        self._chars_used = 0

    @property
    def remaining(self) -> int:
        """Characters remaining in this turn's budget."""
        return max(0, self.max_chars_per_turn - self._chars_used)

    def reset(self) -> None:
        """Reset budget at the start of each LLM round."""
        self._chars_used = 0

    def allocate(self, raw_output: str, max_per_tool: int) -> str:
        """Allocate budget for a tool result, applying truncation if needed.

        Args:
            raw_output: The raw tool output text.
            max_per_tool: Per-tool character limit (existing config).

        Returns:
            Truncated output fitting within both per-tool and remaining budget.
        """
        remaining = self.remaining
        if remaining <= 0:
            return "[Output budget exhausted for this turn]"

        effective_max = min(max_per_tool, remaining)
        result = truncate_tool_output(raw_output, effective_max)
        self._chars_used += len(result)
        return result
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_turn_budget.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/turn_budget.py tests/test_core/test_turn_budget.py
git commit -m "feat(output-budget): add per-turn output budget tracker

TurnBudgetTracker caps total tool output chars per LLM round.
Prevents context explosion on multi-tool turns (5 calls × 8K → 30K cap).
Graceful degradation when budget exhausted."
```

---

## Task 4: Per-Turn Output Budget — Wire into AgentLoop

**Files:**
- Modify: `src/norn/core/agent.py:92-155` (both `_run_impl` and `run_stream`)
- Modify: `src/norn/core/config.py:104-108`
- Test: `tests/test_core/test_turn_budget.py` (add integration test)

**Step 1: Write the failing test**

```python
# Add to tests/test_core/test_turn_budget.py

class TestAgentIntegration:
    """Test budget tracker integration with AgentLoop."""

    @pytest.mark.asyncio
    async def test_agent_respects_turn_budget(self):
        """AgentLoop uses turn budget to cap multi-tool output."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from norn.core.agent import AgentLoop
        from norn.core.models import LLMResponse, ToolCall

        mock_llm = AsyncMock()
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        # First LLM call returns 3 tool calls, second returns text
        tool_calls = [
            ToolCall(id="t1", name="bash", arguments={"command": "echo hi"}),
            ToolCall(id="t2", name="bash", arguments={"command": "echo bye"}),
            ToolCall(id="t3", name="bash", arguments={"command": "echo end"}),
        ]
        mock_llm.complete = AsyncMock(
            side_effect=[
                LLMResponse(content=None, tool_calls=tool_calls),
                LLMResponse(content="Done"),
            ]
        )

        # Each tool returns 5000 chars
        large_output = "x" * 5000

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            max_turn_output_chars=8000,  # Budget: 8000 total
            max_tool_result_chars=6000,  # Per-tool: 6000
            env_bootstrap=False,
        )

        # Mock tool execution
        with patch.object(loop, "_execute_tool", return_value=MagicMock(output=large_output, error=None)):
            result = await loop.run("test")

        # With 8K budget and 3 tools each producing 5K:
        # Tool 1: gets 5000 (under per-tool 6K and remaining 8K)
        # Tool 2: remaining is 3000, gets truncated to 3000
        # Tool 3: remaining is 0, gets "budget exhausted" marker
        # Verify messages contain budget exhausted marker
        budget_msgs = [m for m in loop.history if m.content and "budget exhausted" in m.content.lower()]
        assert len(budget_msgs) >= 1
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_core/test_turn_budget.py::TestAgentIntegration -v`
Expected: FAIL with `TypeError: AgentLoop.__init__() got an unexpected keyword argument 'max_turn_output_chars'`

**Step 3: Modify AgentLoop and AgentConfig**

In `src/norn/core/config.py`, add to `AgentConfig`:
```python
class AgentConfig(BaseModel):
    max_tool_rounds: int = 25
    minify_tool_schemas: bool = True
    max_tool_result_chars: int = 8000
    max_turn_output_chars: int = 30000  # Phase 10: per-turn budget
    env_bootstrap: bool = True
```

In `src/norn/core/agent.py`:
- Add `max_turn_output_chars: int = 30000` to `__init__`
- Import and instantiate `TurnBudgetTracker`
- In `_run_impl` and `run_stream`: call `tracker.reset()` at top of each round
- Replace `truncate_tool_output(raw_content, self._max_tool_result_chars)` with `self._turn_budget.allocate(raw_content, self._max_tool_result_chars)`

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_turn_budget.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: No regressions

**Step 6: Commit**

```bash
git add src/norn/core/agent.py src/norn/core/config.py tests/test_core/test_turn_budget.py configs/default.yaml
git commit -m "feat(output-budget): wire TurnBudgetTracker into AgentLoop

AgentLoop now tracks per-turn output budget (default 30K chars).
Budget resets each LLM round. Progressive truncation when budget
depletes; marker message when exhausted."
```

---

## Task 5: Domain-Aware Routing — Enhanced Scoring

**Files:**
- Modify: `src/norn/core/router.py:32-65`
- Test: `tests/test_core/test_router_domain.py` (new)

**Step 1: Write the failing tests**

```python
# tests/test_core/test_router_domain.py
"""Tests for domain-aware routing signals (Phase 10)."""

from __future__ import annotations

import pytest

from norn.core.models import Message, Role
from norn.core.router import Tier, _score_complexity


def _msg(content: str) -> list[Message]:
    return [Message(role=Role.USER, content=content)]


class TestDomainRouting:
    """Test domain-aware routing signals."""

    def test_architecture_domain_boosts_score(self):
        """Architecture keywords increase complexity score."""
        msgs = _msg("Design a microservice architecture for the payment system")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True
        assert score >= 2  # domain boost adds +2

    def test_security_domain_boosts_score(self):
        """Security keywords increase complexity score."""
        msgs = _msg("Check for SQL injection vulnerabilities in the auth module")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True
        assert score >= 2

    def test_simple_question_reduces_score(self):
        """Simple questions get a score reduction."""
        msgs = _msg("What is the Python version?")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("simple_pattern", False) is True
        # Simple pattern reduces score, should remain FAST-eligible
        assert score <= 1

    def test_ml_training_boosts_score(self):
        """ML training keywords boost score."""
        msgs = _msg("Fix the training loop gradient accumulation bug")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True

    def test_no_domain_match_unchanged(self):
        """Regular messages without domain keywords have normal scoring."""
        msgs = _msg("Add a print statement to line 42")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is False
        assert signals.get("simple_pattern", False) is False

    def test_debugging_domain_boosts(self):
        """Debugging keywords boost score."""
        msgs = _msg("There's a race condition in the connection pool")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True

    def test_domain_routing_disabled_via_config(self):
        """When domain_routing=False, no domain signals applied."""
        msgs = _msg("Design a microservice architecture")
        score, signals = _score_complexity(msgs, [], domain_routing=False)
        assert "domain_boost" not in signals or signals["domain_boost"] is False
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_router_domain.py -v`
Expected: FAIL (signature mismatch or missing signals)

**Step 3: Modify `router.py`**

Add domain patterns and update `_score_complexity()`:

```python
# Add after _KEYWORD_RE (line 35)
_DOMAIN_BOOST_PATTERNS = {
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
```

Update `_score_complexity` signature to accept `domain_routing: bool = True`:

```python
def _score_complexity(
    messages: list[Message],
    tools: list[dict],
    *,
    domain_routing: bool = True,
) -> tuple[int, dict[str, bool]]:
    last_content = messages[-1].content if messages else None
    long_prompt = bool(last_content and len(last_content) > _LONG_PROMPT_CHARS)
    complex_keyword = bool(last_content and _KEYWORD_RE.search(last_content))
    long_history = len(messages) > _LONG_HISTORY_TURNS
    many_tools = len(tools) > _MANY_TOOLS

    signals: dict[str, bool] = {
        "long_prompt": long_prompt,
        "complex_keyword": complex_keyword,
        "long_history": long_history,
        "many_tools": many_tools,
    }
    score = sum(1 for k, v in signals.items() if v and k != "simple_pattern")

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
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_router_domain.py -v`
Expected: All 7 tests PASS

**Step 5: Run full test suite (especially existing router tests)**

Run: `uv run pytest tests/test_core/ --tb=short -q`
Expected: No regressions

**Step 6: Commit**

```bash
git add src/norn/core/router.py tests/test_core/test_router_domain.py
git commit -m "feat(router): add domain-aware routing signals

Architecture, security, ML training, and debugging keywords now boost
complexity score (+2). Simple question pattern reduces score (-1).
Configurable via domain_routing kwarg. No existing behavior changed
when domains don't match."
```

---

## Task 6: Sliding Window — Context Manager Core

**Files:**
- Create: `src/norn/core/context.py`
- Test: `tests/test_core/test_context_manager.py`

**Step 1: Write the failing tests**

```python
# tests/test_core/test_context_manager.py
"""Tests for sliding window context manager (Phase 10)."""

from __future__ import annotations

import pytest

from norn.core.context import ContextManager
from norn.core.models import Message, Role


def _make_history(n_turns: int, chars_per_msg: int = 100) -> list[Message]:
    """Create a fake conversation history with n user+assistant turns."""
    history: list[Message] = []
    for i in range(n_turns):
        history.append(Message(role=Role.USER, content=f"Question {i}: {'x' * chars_per_msg}"))
        history.append(Message(role=Role.ASSISTANT, content=f"Answer {i}: {'y' * chars_per_msg}"))
    return history


class TestContextManager:
    """Test ContextManager sliding window behavior."""

    @pytest.mark.asyncio
    async def test_short_history_unchanged(self):
        """History under budget is returned verbatim."""
        cm = ContextManager(max_history_tokens=50000, recent_turns_keep=6)
        history = _make_history(3)  # 6 messages, well under budget
        messages = await cm.build_messages("System prompt", history)
        # system + all history messages
        assert len(messages) == 1 + len(history)
        assert messages[0].role == Role.SYSTEM

    @pytest.mark.asyncio
    async def test_long_history_triggers_window(self):
        """History over budget gets windowed."""
        # Very small budget to force windowing
        cm = ContextManager(
            max_history_tokens=500,  # tiny budget
            recent_turns_keep=2,
        )
        history = _make_history(20, chars_per_msg=200)  # way over budget
        messages = await cm.build_messages("System prompt", history)
        # Should have: system + summary + last 4 messages (2 turns × 2 msgs each)
        assert len(messages) <= 1 + 1 + 4 + 1  # system + summary + recent + margin
        # Summary message should exist
        summaries = [m for m in messages if m.role == Role.SYSTEM and "summary" in (m.content or "").lower()]
        # Or check for a dedicated summary message
        assert any("Summary" in (m.content or "") or "summary" in (m.content or "").lower() for m in messages)

    @pytest.mark.asyncio
    async def test_recent_turns_always_kept(self):
        """The last N turns are always preserved verbatim."""
        cm = ContextManager(max_history_tokens=100, recent_turns_keep=3)
        history = _make_history(10, chars_per_msg=50)
        messages = await cm.build_messages("System", history)
        # Last 3 turns = last 6 messages from history
        last_6 = history[-6:]
        # They should appear verbatim at the end
        for orig in last_6:
            assert any(m.content == orig.content for m in messages)

    @pytest.mark.asyncio
    async def test_summary_cached(self):
        """Summary is cached and not regenerated every call."""
        cm = ContextManager(max_history_tokens=200, recent_turns_keep=2)
        history = _make_history(10, chars_per_msg=100)
        # First call generates summary
        await cm.build_messages("System", history)
        first_summary = cm._cached_summary
        # Second call with same history uses cache
        await cm.build_messages("System", history)
        assert cm._cached_summary is first_summary  # same object

    @pytest.mark.asyncio
    async def test_disabled_returns_full_history(self):
        """When sliding_window=False, returns full history."""
        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2, enabled=False)
        history = _make_history(20, chars_per_msg=200)
        messages = await cm.build_messages("System", history)
        assert len(messages) == 1 + len(history)  # system + all

    def test_token_estimate(self):
        """Token estimation works without litellm."""
        cm = ContextManager(max_history_tokens=1000, recent_turns_keep=2)
        msgs = _make_history(5, chars_per_msg=100)
        tokens = cm._estimate_tokens(msgs)
        # ~100 chars per msg × 10 msgs / 4 chars per token ≈ 250 tokens
        assert 200 < tokens < 400
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_context_manager.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'norn.core.context'`

**Step 3: Write the implementation**

```python
# src/norn/core/context.py
"""Sliding window context manager with history summarization (Phase 10).

Manages conversation history to keep total context within a token budget.
Older turns are evicted and replaced with a condensed summary. Recent
turns are always preserved verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from norn.core.models import Message, Role

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider

# Default summarization prompt
_SUMMARY_PROMPT = (
    "Summarize the conversation below in ≤{max_tokens} tokens.\n"
    "Preserve: key decisions, file paths mentioned, task context, errors encountered.\n"
    "Omit: verbose tool outputs, intermediate steps that led nowhere.\n"
    "Format: dense bullet points.\n\n"
    "{conversation}"
)


class ContextManager:
    """Manages conversation history with sliding window and summarization.

    When total history exceeds max_history_tokens, older messages are
    evicted and replaced with a summary. The last `recent_turns_keep`
    turns are always preserved verbatim.
    """

    def __init__(
        self,
        max_history_tokens: int = 8000,
        recent_turns_keep: int = 6,
        summary_max_tokens: int = 300,
        summary_provider: LLMProvider | None = None,
        enabled: bool = True,
    ) -> None:
        self.max_history_tokens = max_history_tokens
        self.recent_turns_keep = recent_turns_keep
        self.summary_max_tokens = summary_max_tokens
        self._summary_provider = summary_provider
        self.enabled = enabled
        # Cache
        self._cached_summary: str | None = None
        self._cached_history_len: int = 0  # len(history) when summary was generated

    async def build_messages(
        self,
        system_prompt: str,
        history: list[Message],
    ) -> list[Message]:
        """Build message list with optional sliding window.

        Returns: [system, summary_msg?, ...recent_turns]
        """
        system_msg = Message(role=Role.SYSTEM, content=system_prompt)

        if not self.enabled or not history:
            return [system_msg, *history]

        total_tokens = self._estimate_tokens(history)
        if total_tokens <= self.max_history_tokens:
            return [system_msg, *history]

        # Need to window: keep last N turns
        recent_msg_count = self.recent_turns_keep * 2  # user + assistant per turn
        recent_msg_count = min(recent_msg_count, len(history))

        recent = history[-recent_msg_count:] if recent_msg_count > 0 else []
        old = history[:-recent_msg_count] if recent_msg_count < len(history) else []

        # Generate or use cached summary
        summary = await self._get_summary(old, len(history))

        messages = [system_msg]
        if summary:
            messages.append(Message(role=Role.SYSTEM, content=f"## Conversation Summary\n\n{summary}"))
        messages.extend(recent)
        return messages

    async def _get_summary(self, old_messages: list[Message], history_len: int) -> str | None:
        """Get summary of old messages, using cache when possible."""
        if not old_messages:
            return None

        # Use cache if history hasn't grown
        if self._cached_summary and self._cached_history_len == history_len:
            return self._cached_summary

        summary = await self._summarize(old_messages)
        self._cached_summary = summary
        self._cached_history_len = history_len
        return summary

    async def _summarize(self, messages: list[Message]) -> str:
        """Generate a condensed summary of messages.

        If no summary_provider is set, uses a simple extractive fallback.
        """
        if self._summary_provider is not None:
            return await self._llm_summarize(messages)
        return self._extractive_summary(messages)

    async def _llm_summarize(self, messages: list[Message]) -> str:
        """Use an LLM to generate the summary."""
        conversation = "\n".join(
            f"{m.role.value}: {(m.content or '')[:200]}" for m in messages
        )
        prompt = _SUMMARY_PROMPT.format(
            max_tokens=self.summary_max_tokens,
            conversation=conversation,
        )
        response = await self._summary_provider.complete(  # type: ignore[union-attr]
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            max_tokens=self.summary_max_tokens,
        )
        return response.content or ""

    def _extractive_summary(self, messages: list[Message]) -> str:
        """Simple extractive summary when no LLM is available.

        Keeps first user message + key assistant messages (non-tool).
        """
        parts: list[str] = []
        for msg in messages:
            if msg.role == Role.USER and msg.content:
                # Keep first 100 chars of each user message
                parts.append(f"- User: {msg.content[:100]}")
            elif msg.role == Role.ASSISTANT and msg.content and len(msg.content) > 20:
                # Keep first 80 chars of substantive assistant messages
                parts.append(f"- Assistant: {msg.content[:80]}")
            # Skip tool messages in extractive mode
            if len(parts) >= 10:
                break
        return "\n".join(parts) if parts else "No prior context."

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.

        Uses chars/4 heuristic. For production, integrate litellm.token_counter().
        """
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // 4
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_context_manager.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/context.py tests/test_core/test_context_manager.py
git commit -m "feat(context): add sliding window context manager

ContextManager implements token-budgeted history with summarization.
Older turns are evicted and replaced with a summary (extractive
fallback or LLM-generated). Recent N turns always preserved verbatim.
Summary is cached until new eviction is needed."
```

---

## Task 7: Sliding Window — Wire into AgentLoop

**Files:**
- Modify: `src/norn/core/agent.py:44-99,157-170`
- Modify: `src/norn/core/config.py` (add `ContextConfig`)
- Modify: `src/norn/cli/main.py` (pass context_manager to AgentLoop)
- Test: `tests/test_core/test_context_manager.py` (add agent integration test)

**Step 1: Write the failing test**

```python
# Add to tests/test_core/test_context_manager.py

class TestAgentIntegration:
    """Test context manager integration with AgentLoop."""

    @pytest.mark.asyncio
    async def test_agent_uses_context_manager(self):
        """AgentLoop with context_manager uses windowed messages."""
        from unittest.mock import AsyncMock, MagicMock

        from norn.core.agent import AgentLoop
        from norn.core.context import ContextManager
        from norn.core.models import LLMResponse

        mock_llm = AsyncMock()
        mock_llm.complete = AsyncMock(return_value=LLMResponse(content="ok"))
        mock_registry = MagicMock()
        mock_registry.get_schemas.return_value = []

        cm = ContextManager(max_history_tokens=100, recent_turns_keep=2, enabled=True)

        loop = AgentLoop(
            llm=mock_llm,
            registry=mock_registry,
            cwd="/tmp",
            context_manager=cm,
            env_bootstrap=False,
        )

        # Simulate many prior turns
        for i in range(20):
            loop.history.append(Message(role=Role.USER, content=f"Q{i}: {'x' * 200}"))
            loop.history.append(Message(role=Role.ASSISTANT, content=f"A{i}: {'y' * 200}"))

        # Next turn should use windowed context
        await loop.run("final question")

        # Verify the messages sent to LLM are windowed (not 41+ messages)
        call_args = mock_llm.complete.call_args
        messages_sent = call_args.kwargs.get("messages") or call_args[0][0]
        assert len(messages_sent) < 10  # windowed, not full 43 messages
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest "tests/test_core/test_context_manager.py::TestAgentIntegration" -v`
Expected: FAIL with `TypeError: AgentLoop.__init__() got an unexpected keyword argument 'context_manager'`

**Step 3: Modify AgentLoop**

Add `context_manager: ContextManager | None = None` to `__init__`. In `_run_impl` and `run_stream`, replace the message construction:

```python
# Before:
messages = [Message(role=Role.SYSTEM, content=self._build_system_prompt()), *self.history]

# After:
if self._context_manager is not None:
    messages = await self._context_manager.build_messages(
        self._build_system_prompt(), self.history
    )
else:
    messages = [Message(role=Role.SYSTEM, content=self._build_system_prompt()), *self.history]
```

Add `ContextConfig` to `config.py`:
```python
class ContextConfig(BaseModel):
    sliding_window: bool = False  # opt-in
    max_history_tokens: int = 8000
    recent_turns_keep: int = 6
    summary_max_tokens: int = 300
    summary_model: str | None = None
```

Add to `NornConfig`:
```python
context: ContextConfig = ContextConfig()
```

Wire in `cli/main.py` where `AgentLoop` is constructed.

**Step 4: Run tests**

Run: `uv run pytest tests/test_core/test_context_manager.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: No regressions

**Step 6: Commit**

```bash
git add src/norn/core/agent.py src/norn/core/config.py src/norn/cli/main.py tests/test_core/test_context_manager.py configs/default.yaml
git commit -m "feat(context): wire ContextManager into AgentLoop

AgentLoop accepts optional context_manager for sliding window behavior.
When enabled, messages are windowed before each LLM call. Old turns
are summarized, recent N turns kept verbatim. Opt-in via config:
context.sliding_window=true."
```

---

## Task 8: Config + CLI Wiring + Final Integration

**Files:**
- Modify: `src/norn/cli/main.py` (pass all new config to AgentLoop)
- Modify: `configs/default.yaml` (document all new settings)
- Test: run full suite + manual smoke test

**Step 1: Update `configs/default.yaml`**

Add Phase 10 settings with documentation comments:

```yaml
# Phase 10 — Harness Engineering
agent:
  max_tool_rounds: 25
  minify_tool_schemas: true
  max_tool_result_chars: 8000
  max_turn_output_chars: 30000   # Per-turn budget across all tool calls
  env_bootstrap: true            # Inject [Environment] snapshot into system prompt

context:
  sliding_window: false          # Opt-in: evict old turns when over budget
  max_history_tokens: 8000       # Token budget for conversation history
  recent_turns_keep: 6           # Always keep last N turns verbatim
  summary_max_tokens: 300        # Max tokens for summary of evicted turns
  summary_model: null            # null = use primary model for summarization

router:
  domain_routing: true           # Enable domain-aware routing signals
```

**Step 2: Wire new config into CLI main.py**

In the function that builds `AgentLoop` (wherever the loop is instantiated), pass:
- `env_bootstrap=config.agent.env_bootstrap`
- `max_turn_output_chars=config.agent.max_turn_output_chars`
- `context_manager=ContextManager(...)` if `config.context.sliding_window`

**Step 3: Run full test suite**

Run: `uv run pytest --tb=short -q`
Expected: All tests pass (560 + new ~25 tests ≈ 585+)

**Step 4: Run ruff**

Run: `uv run ruff check src/ tests/`
Expected: No new errors introduced

**Step 5: Commit**

```bash
git add configs/default.yaml src/norn/cli/main.py
git commit -m "feat(phase10): wire all Phase 10 config into CLI

Complete Phase 10 integration: env_bootstrap, per-turn output budget,
sliding window (opt-in), and domain-aware routing all configurable
via configs/default.yaml."
```

---

## Task 9: Update Tracker + Final Verification

**Files:**
- Modify: `docs/plans/2026-04-21-norn-phase9-observability-followups.md` (add Phase 10 results)

**Step 1: Run full test suite + ruff one final time**

Run: `uv run pytest --tb=short -q && uv run ruff check src/ tests/`
Expected: All green

**Step 2: Update tracker with results**

Add a Phase 10 section to the observability followups tracker documenting:
- Number of tests added
- Features delivered
- Current ruff error count

**Step 3: Commit**

```bash
git add docs/plans/2026-04-21-norn-phase9-observability-followups.md
git commit -m "docs: update tracker with Phase 10 results"
```

---

## Summary

| Task | Feature | Tests Added | Commits |
|------|---------|-------------|---------|
| 1-2 | Env Bootstrap | ~9 | 2 |
| 3-4 | Per-Turn Output Budget | ~7 | 2 |
| 5 | Domain-Aware Routing | ~7 | 1 |
| 6-7 | Sliding Window + Summarization | ~7 | 2 |
| 8 | Config + CLI Wiring | — | 1 |
| 9 | Tracker Update | — | 1 |
| **Total** | **4 features** | **~30** | **9** |

**Estimated effort:** 4-5 sessions
**Risk level:** Low (all modifications are additive, config-gated, with graceful fallback)
