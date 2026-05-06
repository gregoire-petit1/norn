"""Tests for repo map wiring into AgentLoop (W3.1 Task 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

from norn.core.agent import AgentLoop

if TYPE_CHECKING:
    from pathlib import Path


class TestRepoMapWiring:
    """Test that repo map is generated and injected into system prompt."""

    def test_repo_map_injected_in_system_prompt(self, tmp_path: Path):
        """Repo map appears in system prompt when enabled."""
        src = tmp_path / "app.py"
        src.write_text("class Application:\n    def run(self):\n        pass\n")

        llm = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            cwd=str(tmp_path),
            env_bootstrap=False,
            repo_map=True,
            repo_map_max_chars=2000,
            repo_map_languages=["python"],
            repo_map_exclude=[],
        )

        prompt = agent._build_system_prompt()
        assert "## Repo Map" in prompt
        assert "Application" in prompt
        assert "run" in prompt

    def test_repo_map_disabled(self, tmp_path: Path):
        """Repo map not in prompt when disabled."""
        src = tmp_path / "app.py"
        src.write_text("class Application:\n    def run(self):\n        pass\n")

        llm = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            cwd=str(tmp_path),
            env_bootstrap=False,
            repo_map=False,
        )

        prompt = agent._build_system_prompt()
        assert "## Repo Map" not in prompt

    def test_repo_map_empty_dir(self, tmp_path: Path):
        """Repo map handles empty directories gracefully."""
        llm = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            cwd=str(tmp_path),
            env_bootstrap=False,
            repo_map=True,
            repo_map_max_chars=2000,
            repo_map_languages=["python"],
            repo_map_exclude=[],
        )

        # Should not crash; repo_map may be None or empty string
        prompt = agent._build_system_prompt()
        assert isinstance(prompt, str)

    def test_repo_map_respects_max_chars(self, tmp_path: Path):
        """Repo map output respects max_chars limit."""
        for i in range(50):
            (tmp_path / f"mod_{i}.py").write_text(
                f"class Class{i}:\n    def method_a(self):\n        pass\n"
            )

        llm = MagicMock()
        registry = MagicMock()
        registry.get_schemas.return_value = []

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            cwd=str(tmp_path),
            env_bootstrap=False,
            repo_map=True,
            repo_map_max_chars=500,
            repo_map_languages=["python"],
            repo_map_exclude=[],
        )

        assert agent._repo_map is not None
        assert len(agent._repo_map) <= 600  # small margin
