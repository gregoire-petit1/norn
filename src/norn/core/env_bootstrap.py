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

# Language detection: extension -> language name
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

# Package manager detection: file -> manager
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
        if len(rendered) > _MAX_RENDER_CHARS:
            rendered = rendered[: _MAX_RENDER_CHARS - 3] + "..."
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
        dirs = sorted(e for e in entries if (path / e).is_dir() and not e.startswith("."))[:10]
        snap.directory_tree = " ".join(f"{d}/" for d in dirs)
    except OSError:
        pass

    # Python version
    if "python" in snap.languages or any(
        f in snap.key_files for f in ("pyproject.toml", "requirements.txt", "setup.py")
    ):
        snap.python_version = (
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    # Venv detection
    snap.venv_active = bool(os.environ.get("VIRTUAL_ENV")) or (path / ".venv").is_dir()

    return snap
