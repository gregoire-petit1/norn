"""Risk classifier for contextual risk escalation."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class BashClassification:
    """Result of analyzing a bash command."""

    is_destructive: bool = False
    patterns_matched: list[str] = field(default_factory=list)


@dataclass
class EscalationResult:
    """Result of risk escalation analysis."""

    risk: str  # "low", "medium", "high"
    is_destructive: bool = False
    is_protected: bool = False
    has_traversal: bool = False
    reason: str | None = None


# Compiled regex patterns for destructive bash commands
_DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\brm\s+.*-\w*r\w*f\w*\b", re.IGNORECASE), "recursive force delete"),
    (re.compile(r"\brm\s+.*-\w*f\w*r\w*\b", re.IGNORECASE), "recursive force delete"),
    (re.compile(r"\bdrop\s+table\b", re.IGNORECASE), "SQL DROP TABLE"),
    (re.compile(r"\bdelete\s+from\b", re.IGNORECASE), "SQL DELETE FROM"),
    (re.compile(r"\bgit\s+push\s+.*--force\b", re.IGNORECASE), "git force push"),
    (re.compile(r"\bgit\s+push\s+.*-f\b", re.IGNORECASE), "git force push"),
    (re.compile(r"\bmkfs\b", re.IGNORECASE), "format filesystem"),
    (re.compile(r"\bdd\s+if=", re.IGNORECASE), "raw disk write"),
    (re.compile(r":\(\)\s*\{.*\|.*&\s*\}\s*;", re.IGNORECASE), "fork bomb"),
    (re.compile(r">\s*/dev/sd[a-z]", re.IGNORECASE), "raw device overwrite"),
    (re.compile(r"\bchmod\s+.*-\w*R\w*\s+777\s+/", re.IGNORECASE), "recursive chmod 777 root"),
    (re.compile(r"\bchown\s+.*-\w*R\w*\s+\w+\s+/", re.IGNORECASE), "recursive chown root"),
]

_PROTECTED_PATH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"(^|/)\.env($|/)"),
    re.compile(r"(^|/)\.ssh(/|$)"),
    re.compile(r"(^|/)\.gitconfig$"),
    re.compile(r"(^|/)\.bashrc$"),
    re.compile(r"(^|/)\.zshrc$"),
    re.compile(r"(^|/)\.bash_profile$"),
    re.compile(r"\bid_rsa\b"),
    re.compile(r"\bcredentials\b", re.IGNORECASE),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
    re.compile(r"(^|/)(etc/passwd|etc/shadow)$"),
]

# Tools that can modify files (risk escalation applies to these)
_WRITE_TOOLS = {"file_write", "file_edit", "bash"}


class RiskClassifier:
    """Classifies and escalates risk based on tool arguments."""

    def classify_bash(self, command: str) -> BashClassification:
        """Analyze a bash command for destructive patterns."""
        matched = []
        for pattern, description in _DESTRUCTIVE_PATTERNS:
            if pattern.search(command):
                matched.append(description)
        return BashClassification(
            is_destructive=len(matched) > 0,
            patterns_matched=matched,
        )

    def is_protected_path(self, path: str) -> bool:
        """Check if a path is protected (sensitive config/secret file)."""
        # Expand ~ for matching
        expanded = path.replace("~", str(Path.home()))
        return any(pattern.search(expanded) for pattern in _PROTECTED_PATH_PATTERNS)

    def has_path_traversal(self, path: str) -> bool:
        """Detect path traversal attempts."""
        # Check raw path for ../ or ..\\ sequences
        if ".." in path and ("../" in path or "..\\" in path):
            return True
        # Check URL-encoded variants
        decoded = urllib.parse.unquote(path)
        return decoded != path and ".." in decoded and ("../" in decoded or "..\\" in decoded)

    def escalate(
        self,
        tool_name: str,
        base_risk: str,
        arguments: dict[str, Any],
    ) -> EscalationResult:
        """Determine if a tool call's risk should be escalated."""
        result = EscalationResult(risk=base_risk)

        # Read-only tools never get escalated
        if tool_name not in _WRITE_TOOLS:
            return result

        # Check bash commands for destructive patterns
        if tool_name == "bash":
            command = arguments.get("command", "")
            classification = self.classify_bash(command)
            if classification.is_destructive:
                result.is_destructive = True
                result.risk = "high"
                result.reason = (
                    f"Destructive command detected: {', '.join(classification.patterns_matched)}"
                )
                return result

        # Check file paths for protected paths and traversal
        path = arguments.get("path", "")
        if path:
            if self.has_path_traversal(path):
                result.has_traversal = True
                result.risk = "high"
                result.reason = f"Path traversal detected in: {path}"
                return result

            if tool_name in ("file_write", "file_edit") and self.is_protected_path(path):
                result.is_protected = True
                result.risk = "high"
                result.reason = f"Protected path: {path}"
                return result

        return result
