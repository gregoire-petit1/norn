"""Tests for environment bootstrap module."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import patch

from norn.core.env_bootstrap import EnvironmentSnapshot, scan_environment

if TYPE_CHECKING:
    from pathlib import Path


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
        with patch("norn.core.env_bootstrap._run_git") as mock_git:

            def git_side_effect(cmd, cwd):
                if cmd == ["rev-parse", "--abbrev-ref", "HEAD"]:
                    return "main"
                if cmd == ["status", "--porcelain"]:
                    return ""
                if cmd == ["log", "--oneline", "-3"]:
                    return "abc feat: init\ndef fix: typo"
                return ""

            mock_git.side_effect = git_side_effect
            result = scan_environment(str(tmp_path))
        assert result.git_branch == "main"
        assert result.git_dirty is False

    def test_scan_graceful_on_failure(self, tmp_path: Path):
        """scan_environment never raises, even on broken state."""
        result = scan_environment(str(tmp_path))
        assert result.git_branch is None
        assert result.languages == []

    def test_scan_detects_venv(self, tmp_path: Path):
        """Detects virtual environment."""
        (tmp_path / ".venv").mkdir()
        result = scan_environment(str(tmp_path))
        assert result.venv_active is True


class TestAgentIntegration:
    """Test env bootstrap integration with AgentLoop."""

    def test_system_prompt_includes_snapshot(self, tmp_path: Path):
        """When env_bootstrap=True, system prompt contains [Environment] block."""
        from unittest.mock import MagicMock

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
