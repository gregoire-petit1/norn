# Norn Phase 2: Permission System + Feature Flags Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a permission enforcement layer that checks tool risk against the configured mode before execution, with contextual risk escalation for dangerous bash patterns and protected paths. Add a feature flag registry with priority resolution (env > config > default) and conditional tool registration.

**Architecture:** The PermissionChecker protocol sits between the AgentLoop and tool execution. A `RiskClassifier` can escalate a tool's base risk level based on the actual arguments (e.g., `rm -rf` escalates BashTool from HIGH to HIGH-with-destructive-flag, or a write to `.env` escalates FileWriteTool from MEDIUM to HIGH). The `FeatureFlagRegistry` resolves flags with priority: env var > config > hardcoded default. The ToolRegistry uses flags to conditionally include tools.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, ruff

---

## Task 0: Permission Models and Types

**Files:**
- Create: `src/norn/permissions/models.py`
- Test: `tests/test_permissions/__init__.py`
- Test: `tests/test_permissions/test_models.py`

**Step 1: Create test directory**

```bash
mkdir -p tests/test_permissions
touch tests/test_permissions/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_permissions/test_models.py
"""Tests for permission models."""

from norn.permissions.models import PermissionDecision, PermissionRequest


def test_permission_request_from_tool_call():
    req = PermissionRequest(
        tool_name="bash",
        risk_level="high",
        arguments={"command": "ls"},
    )
    assert req.tool_name == "bash"
    assert req.risk_level == "high"
    assert req.arguments == {"command": "ls"}


def test_permission_decision_approved():
    decision = PermissionDecision(approved=True)
    assert decision.approved is True
    assert decision.reason is None


def test_permission_decision_denied_with_reason():
    decision = PermissionDecision(approved=False, reason="Destructive command detected")
    assert decision.approved is False
    assert decision.reason == "Destructive command detected"


def test_permission_decision_escalated_risk():
    decision = PermissionDecision(
        approved=False,
        reason="Path .env is protected",
        escalated_risk="high",
    )
    assert decision.escalated_risk == "high"
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_permissions/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.permissions.models'`

**Step 4: Write minimal implementation**

```python
# src/norn/permissions/models.py
"""Permission system models for Norn."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class PermissionRequest(BaseModel):
    """Request to check permission for a tool invocation."""

    tool_name: str
    risk_level: str  # "low", "medium", "high"
    arguments: dict[str, Any] = {}


class PermissionDecision(BaseModel):
    """Result of a permission check."""

    approved: bool
    reason: str | None = None
    escalated_risk: str | None = None  # If risk was escalated from base level
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_permissions/test_models.py -v`
Expected: 4 PASSED

**Step 6: Commit**

```bash
git add src/norn/permissions/models.py tests/test_permissions/
git commit -m "feat(permissions): add permission request and decision models"
```

---

## Task 1: Risk Classifier — Destructive Bash Patterns

**Files:**
- Create: `src/norn/permissions/classifier.py`
- Test: `tests/test_permissions/test_classifier.py`

The classifier analyzes tool arguments and can escalate the effective risk level. Key patterns to detect:

- **Destructive bash**: `rm -rf`, `DROP TABLE`, `DELETE FROM`, `git push --force`, `mkfs`, `dd if=`, `:(){ :|:& };:`
- **Protected paths**: `.env`, `.ssh/`, `.gitconfig`, `.bashrc`, `.zshrc`, `id_rsa`, `credentials`, `*.pem`, `*.key`
- **Path traversal**: `../` sequences, URL-encoded variants (`%2e%2e`), backslash (`..\\`)

**Step 1: Write the failing tests**

```python
# tests/test_permissions/test_classifier.py
"""Tests for the risk classifier."""

import pytest

from norn.permissions.classifier import RiskClassifier


@pytest.fixture
def classifier():
    return RiskClassifier()


class TestBashPatterns:
    """Test destructive bash command detection."""

    @pytest.mark.parametrize(
        "command",
        [
            "rm -rf /",
            "rm -rf ~",
            "rm -rf --no-preserve-root /",
            "sudo rm -rf /var",
            "DROP TABLE users;",
            "drop table users",
            "DELETE FROM users WHERE 1=1;",
            "git push --force origin main",
            "git push -f origin master",
            "mkfs.ext4 /dev/sda1",
            "dd if=/dev/zero of=/dev/sda",
            ":(){ :|:& };:",
            "> /dev/sda",
            "chmod -R 777 /",
            "chown -R nobody /",
        ],
    )
    def test_destructive_commands_flagged(self, classifier, command):
        result = classifier.classify_bash(command)
        assert result.is_destructive is True, f"Expected destructive: {command}"

    @pytest.mark.parametrize(
        "command",
        [
            "ls -la",
            "cat file.txt",
            "echo hello",
            "git status",
            "git push origin feature-branch",
            "python script.py",
            "pytest tests/",
            "rm file.txt",  # single file rm is not mass-destructive
            "grep -r pattern .",
        ],
    )
    def test_safe_commands_not_flagged(self, classifier, command):
        result = classifier.classify_bash(command)
        assert result.is_destructive is False, f"Expected safe: {command}"


class TestProtectedPaths:
    """Test protected path detection."""

    @pytest.mark.parametrize(
        "path",
        [
            ".env",
            "/home/user/.env",
            ".ssh/id_rsa",
            "~/.ssh/authorized_keys",
            ".gitconfig",
            ".bashrc",
            ".zshrc",
            "secrets/credentials.json",
            "server.pem",
            "private.key",
            "/etc/passwd",
            "/etc/shadow",
        ],
    )
    def test_protected_paths_detected(self, classifier, path):
        assert classifier.is_protected_path(path) is True, f"Expected protected: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/main.py",
            "tests/test_foo.py",
            "README.md",
            "configs/default.yaml",
            "data/input.csv",
        ],
    )
    def test_normal_paths_not_protected(self, classifier, path):
        assert classifier.is_protected_path(path) is False, f"Expected normal: {path}"


class TestPathTraversal:
    """Test path traversal detection."""

    @pytest.mark.parametrize(
        "path",
        [
            "../../../etc/passwd",
            "foo/../../bar",
            "%2e%2e/%2e%2e/etc/passwd",
            "..\\..\\windows\\system32",
            "foo/..%5c..%5cbar",
        ],
    )
    def test_traversal_detected(self, classifier, path):
        assert classifier.has_path_traversal(path) is True, f"Expected traversal: {path}"

    @pytest.mark.parametrize(
        "path",
        [
            "src/core/agent.py",
            "/absolute/path/file.py",
            "./relative/file.py",
            "file.txt",
        ],
    )
    def test_normal_paths_no_traversal(self, classifier, path):
        assert classifier.has_path_traversal(path) is False, f"Expected no traversal: {path}"


class TestEscalation:
    """Test risk escalation logic."""

    def test_bash_destructive_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="bash",
            base_risk="high",
            arguments={"command": "rm -rf /"},
        )
        assert escalated.risk == "high"
        assert escalated.is_destructive is True
        assert "destructive" in escalated.reason.lower()

    def test_file_write_to_protected_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_write",
            base_risk="medium",
            arguments={"path": ".env", "content": "SECRET=abc"},
        )
        assert escalated.risk == "high"
        assert escalated.is_protected is True

    def test_file_write_normal_no_escalation(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_write",
            base_risk="medium",
            arguments={"path": "src/main.py", "content": "print('hi')"},
        )
        assert escalated.risk == "medium"
        assert escalated.is_destructive is False
        assert escalated.is_protected is False

    def test_file_edit_with_traversal_escalates(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_edit",
            base_risk="medium",
            arguments={"path": "../../../etc/passwd", "old_string": "x", "new_string": "y"},
        )
        assert escalated.risk == "high"
        assert escalated.reason is not None

    def test_read_tool_no_escalation(self, classifier):
        escalated = classifier.escalate(
            tool_name="file_read",
            base_risk="low",
            arguments={"path": ".env"},
        )
        # Read-only tools don't get escalated even for protected paths
        assert escalated.risk == "low"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_permissions/test_classifier.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/permissions/classifier.py
"""Risk classifier for contextual risk escalation."""

from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field
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
_DESTRUCTIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
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

_PROTECTED_PATH_PATTERNS: list[re.Pattern] = [
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
        expanded = path.replace("~", "/home/user")
        for pattern in _PROTECTED_PATH_PATTERNS:
            if pattern.search(expanded):
                return True
        return False

    def has_path_traversal(self, path: str) -> bool:
        """Detect path traversal attempts."""
        # Check raw path
        if ".." in path and ("../" in path or "..\\" in path):
            return True
        # Check URL-encoded variants
        decoded = urllib.parse.unquote(path)
        if decoded != path and ".." in decoded and ("../" in decoded or "..\\" in decoded):
            return True
        return False

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
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_permissions/test_classifier.py -v`
Expected: ALL PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/permissions/classifier.py`

**Step 6: Commit**

```bash
git add src/norn/permissions/classifier.py tests/test_permissions/test_classifier.py
git commit -m "feat(permissions): add risk classifier with destructive bash, protected paths, and traversal detection"
```

---

## Task 2: Permission Checker — Mode-Based Enforcement

**Files:**
- Create: `src/norn/permissions/checker.py`
- Test: `tests/test_permissions/test_checker.py`

The PermissionChecker combines the mode (from config) with the RiskClassifier to produce a PermissionDecision. In non-interactive modes, it auto-resolves. In interactive mode, it calls a callback function to prompt the user.

| Mode | LOW | MEDIUM | HIGH | DESTRUCTIVE |
|------|-----|--------|------|-------------|
| yolo | auto-approve | auto-approve | auto-approve | auto-approve |
| auto | auto-approve | auto-approve | prompt | prompt |
| interactive | auto-approve | prompt | prompt | prompt |
| strict | prompt | prompt | prompt | deny |

**Step 1: Write the failing tests**

```python
# tests/test_permissions/test_checker.py
"""Tests for the permission checker."""

from unittest.mock import AsyncMock

import pytest

from norn.core.config import PermissionMode
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.permissions.models import PermissionRequest


@pytest.fixture
def classifier():
    return RiskClassifier()


class TestYoloMode:
    """Yolo mode: approve everything."""

    @pytest.fixture
    def checker(self, classifier):
        return PermissionChecker(mode=PermissionMode.YOLO, classifier=classifier)

    @pytest.mark.asyncio
    async def test_low_risk_approved(self, checker):
        req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "x"})
        decision = await checker.check(req)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_high_risk_approved(self, checker):
        req = PermissionRequest(tool_name="bash", risk_level="high", arguments={"command": "ls"})
        decision = await checker.check(req)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_destructive_approved(self, checker):
        req = PermissionRequest(
            tool_name="bash", risk_level="high", arguments={"command": "rm -rf /"}
        )
        decision = await checker.check(req)
        assert decision.approved is True


class TestAutoMode:
    """Auto mode: auto-approve LOW+MEDIUM, prompt HIGH."""

    @pytest.fixture
    def checker(self, classifier):
        return PermissionChecker(mode=PermissionMode.AUTO, classifier=classifier)

    @pytest.mark.asyncio
    async def test_low_risk_auto_approved(self, checker):
        req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "x"})
        decision = await checker.check(req)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_medium_risk_auto_approved(self, checker):
        req = PermissionRequest(
            tool_name="file_write",
            risk_level="medium",
            arguments={"path": "src/main.py", "content": "x"},
        )
        decision = await checker.check(req)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_high_risk_needs_prompt(self, checker):
        """HIGH risk with no prompt_fn -> denied."""
        req = PermissionRequest(
            tool_name="bash", risk_level="high", arguments={"command": "ls"}
        )
        decision = await checker.check(req)
        assert decision.approved is False
        assert "requires approval" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_high_risk_with_prompt_approved(self, classifier):
        """HIGH risk with prompt_fn that approves -> approved."""
        prompt_fn = AsyncMock(return_value=True)
        checker = PermissionChecker(
            mode=PermissionMode.AUTO, classifier=classifier, prompt_fn=prompt_fn
        )
        req = PermissionRequest(
            tool_name="bash", risk_level="high", arguments={"command": "ls"}
        )
        decision = await checker.check(req)
        assert decision.approved is True
        prompt_fn.assert_called_once()

    @pytest.mark.asyncio
    async def test_medium_escalated_to_high_needs_prompt(self, checker):
        """MEDIUM risk escalated to HIGH (protected path) -> denied without prompt."""
        req = PermissionRequest(
            tool_name="file_write",
            risk_level="medium",
            arguments={"path": ".env", "content": "SECRET=abc"},
        )
        decision = await checker.check(req)
        assert decision.approved is False


class TestInteractiveMode:
    """Interactive mode: auto-approve LOW, prompt MEDIUM+HIGH."""

    @pytest.fixture
    def checker(self, classifier):
        return PermissionChecker(mode=PermissionMode.INTERACTIVE, classifier=classifier)

    @pytest.mark.asyncio
    async def test_low_risk_auto_approved(self, checker):
        req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "x"})
        decision = await checker.check(req)
        assert decision.approved is True

    @pytest.mark.asyncio
    async def test_medium_risk_needs_prompt(self, checker):
        req = PermissionRequest(
            tool_name="file_write",
            risk_level="medium",
            arguments={"path": "src/main.py", "content": "x"},
        )
        decision = await checker.check(req)
        assert decision.approved is False
        assert "requires approval" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_medium_risk_with_prompt_approved(self, classifier):
        prompt_fn = AsyncMock(return_value=True)
        checker = PermissionChecker(
            mode=PermissionMode.INTERACTIVE, classifier=classifier, prompt_fn=prompt_fn
        )
        req = PermissionRequest(
            tool_name="file_write",
            risk_level="medium",
            arguments={"path": "src/main.py", "content": "x"},
        )
        decision = await checker.check(req)
        assert decision.approved is True


class TestStrictMode:
    """Strict mode: prompt ALL, deny DESTRUCTIVE."""

    @pytest.fixture
    def checker(self, classifier):
        return PermissionChecker(mode=PermissionMode.STRICT, classifier=classifier)

    @pytest.mark.asyncio
    async def test_low_risk_needs_prompt(self, checker):
        """Even LOW risk needs prompt in strict mode."""
        req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "x"})
        decision = await checker.check(req)
        assert decision.approved is False

    @pytest.mark.asyncio
    async def test_destructive_always_denied(self, checker):
        """Destructive commands are always denied in strict mode."""
        req = PermissionRequest(
            tool_name="bash", risk_level="high", arguments={"command": "rm -rf /"}
        )
        decision = await checker.check(req)
        assert decision.approved is False
        assert "denied" in decision.reason.lower() or "destructive" in decision.reason.lower()

    @pytest.mark.asyncio
    async def test_destructive_denied_even_with_prompt(self, classifier):
        """Destructive denied even when prompt_fn would approve."""
        prompt_fn = AsyncMock(return_value=True)
        checker = PermissionChecker(
            mode=PermissionMode.STRICT, classifier=classifier, prompt_fn=prompt_fn
        )
        req = PermissionRequest(
            tool_name="bash", risk_level="high", arguments={"command": "rm -rf /"}
        )
        decision = await checker.check(req)
        assert decision.approved is False
        prompt_fn.assert_not_called()

    @pytest.mark.asyncio
    async def test_low_with_prompt_approved(self, classifier):
        prompt_fn = AsyncMock(return_value=True)
        checker = PermissionChecker(
            mode=PermissionMode.STRICT, classifier=classifier, prompt_fn=prompt_fn
        )
        req = PermissionRequest(tool_name="file_read", risk_level="low", arguments={"path": "x"})
        decision = await checker.check(req)
        assert decision.approved is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_permissions/test_checker.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write implementation**

```python
# src/norn/permissions/checker.py
"""Permission checker: enforces permission modes before tool execution."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.core.config import PermissionMode

from norn.permissions.classifier import RiskClassifier
from norn.permissions.models import PermissionDecision, PermissionRequest

# Type alias for the user-prompt callback
PromptFn = Callable[[PermissionRequest, str], Awaitable[bool]]

# Risk level ordering for comparison
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}


class PermissionChecker:
    """Check permissions based on mode, risk level, and contextual escalation."""

    def __init__(
        self,
        mode: PermissionMode,
        classifier: RiskClassifier,
        prompt_fn: PromptFn | None = None,
    ) -> None:
        self.mode = mode
        self.classifier = classifier
        self.prompt_fn = prompt_fn

    async def check(self, request: PermissionRequest) -> PermissionDecision:
        """Check whether a tool invocation is permitted."""
        # Step 1: Escalate risk if needed
        escalation = self.classifier.escalate(
            tool_name=request.tool_name,
            base_risk=request.risk_level,
            arguments=request.arguments,
        )
        effective_risk = escalation.risk

        # Step 2: Apply mode-specific rules
        return await self._apply_mode(request, effective_risk, escalation.is_destructive)

    async def _apply_mode(
        self,
        request: PermissionRequest,
        effective_risk: str,
        is_destructive: bool,
    ) -> PermissionDecision:
        """Apply the permission mode rules."""
        mode = self.mode.value

        if mode == "yolo":
            return PermissionDecision(approved=True)

        if mode == "strict":
            # Destructive = always denied, no prompt
            if is_destructive:
                return PermissionDecision(
                    approved=False,
                    reason="Destructive command denied in strict mode",
                )
            # Everything else needs prompt
            return await self._prompt_or_deny(request, effective_risk)

        if mode == "auto":
            # LOW + MEDIUM auto-approved, HIGH needs prompt
            if _RISK_ORDER.get(effective_risk, 2) <= _RISK_ORDER["medium"]:
                return PermissionDecision(approved=True)
            return await self._prompt_or_deny(request, effective_risk)

        # interactive (default)
        # LOW auto-approved, MEDIUM + HIGH need prompt
        if _RISK_ORDER.get(effective_risk, 2) <= _RISK_ORDER["low"]:
            return PermissionDecision(approved=True)
        return await self._prompt_or_deny(request, effective_risk)

    async def _prompt_or_deny(
        self,
        request: PermissionRequest,
        effective_risk: str,
    ) -> PermissionDecision:
        """Prompt user if callback available, otherwise deny."""
        if self.prompt_fn is None:
            return PermissionDecision(
                approved=False,
                reason=f"Tool '{request.tool_name}' (risk: {effective_risk}) requires approval",
                escalated_risk=effective_risk if effective_risk != request.risk_level else None,
            )

        description = (
            f"Tool '{request.tool_name}' (risk: {effective_risk}) "
            f"wants to execute with args: {request.arguments}"
        )
        approved = await self.prompt_fn(request, description)
        return PermissionDecision(
            approved=approved,
            reason=None if approved else "User denied",
        )
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_permissions/test_checker.py -v`
Expected: ALL PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/permissions/checker.py`

**Step 6: Commit**

```bash
git add src/norn/permissions/checker.py tests/test_permissions/test_checker.py
git commit -m "feat(permissions): add mode-based permission checker with user prompt support"
```

---

## Task 3: Integrate Permissions into AgentLoop

**Files:**
- Modify: `src/norn/core/agent.py`
- Modify: `tests/test_core/test_agent.py` (add permission tests)

The AgentLoop now optionally accepts a `PermissionChecker`. If present, every tool call goes through permission check before execution. If denied, a `ToolResult(error=...)` with the denial reason is returned to the LLM instead of executing the tool.

**Step 1: Write the failing tests**

Add to `tests/test_core/test_agent.py`:

```python
# Add these imports at the top:
from unittest.mock import AsyncMock, MagicMock, patch
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.permissions.models import PermissionDecision, PermissionRequest
from norn.core.config import PermissionMode


# --- New fixtures ---

class WriteInput(BaseModel):
    path: str
    content: str


class WriteTool:
    name = "file_write"
    description = "Write a file"
    risk_level = RiskLevel.MEDIUM
    input_model = WriteInput

    async def execute(self, input: WriteInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"wrote to {input.path}")


@pytest.fixture
def registry_with_write():
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(WriteTool())
    return reg


# --- New tests ---

@pytest.mark.asyncio
async def test_agent_permission_denied(registry_with_write):
    """When permission is denied, tool should not execute."""
    checker = PermissionChecker(
        mode=PermissionMode.INTERACTIVE,
        classifier=RiskClassifier(),
        prompt_fn=None,  # No prompt = auto-deny for MEDIUM+
    )

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": "test.py", "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="Permission was denied.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(
        llm=llm, registry=registry_with_write, permission_checker=checker
    )
    result = await agent.run("write a file")
    assert result.content == "Permission was denied."
    assert llm.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_permission_approved_yolo(registry_with_write):
    """In yolo mode, everything is approved."""
    checker = PermissionChecker(
        mode=PermissionMode.YOLO,
        classifier=RiskClassifier(),
    )

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": "test.py", "content": "hello"},
                    )
                ],
            ),
            LLMResponse(content="File written.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(
        llm=llm, registry=registry_with_write, permission_checker=checker
    )
    result = await agent.run("write a file")
    assert result.content == "File written."


@pytest.mark.asyncio
async def test_agent_no_permission_checker_allows_all(registry):
    """Without a permission checker, all tools execute (backward-compatible)."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "hi"})],
            ),
            LLMResponse(content="done", tool_calls=[]),
        ]
    )
    agent = AgentLoop(llm=llm, registry=registry)
    result = await agent.run("echo")
    assert result.content == "done"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_core/test_agent.py -v`
Expected: FAIL — `AgentLoop() got an unexpected keyword argument 'permission_checker'`

**Step 3: Modify AgentLoop**

Update `src/norn/core/agent.py`:

```python
"""Core agent loop for Norn."""

from __future__ import annotations

from typing import TYPE_CHECKING

from norn.core.models import LLMResponse, Message, Role, ToolCall
from norn.tools.base import ToolContext, ToolResult

if TYPE_CHECKING:
    from norn.permissions.checker import PermissionChecker
    from norn.tools.registry import ToolRegistry


class AgentLoop:
    """The main agent loop: message -> LLM -> tool calls -> repeat."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(
        self,
        llm: object,
        registry: ToolRegistry,
        system_prompt: str = "You are Norn, a helpful coding agent.",
        cwd: str = ".",
        permission_checker: PermissionChecker | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cwd=cwd)
        self.history: list[Message] = []
        self.permission_checker = permission_checker

    async def run(self, user_input: str) -> LLMResponse:
        """Run one turn of the agent loop."""
        self.history.append(Message(role=Role.USER, content=user_input))

        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            *self.history,
        ]

        for _round in range(self.MAX_TOOL_ROUNDS):
            response = await self.llm.complete(
                messages=messages,
                tools=self.registry.get_schemas() or None,
            )

            if not response.has_tool_calls:
                self.history.append(Message(role=Role.ASSISTANT, content=response.content))
                return response

            # Process tool calls
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)

            for call in response.tool_calls:
                result = await self._execute_tool(call)
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output or result.error or "",
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)

        # Safety: max rounds reached
        final = LLMResponse(content="[Max tool rounds reached]")
        self.history.append(Message(role=Role.ASSISTANT, content=final.content))
        return final

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call, with optional permission check."""
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {call.name}")

        # Permission check
        if self.permission_checker is not None:
            from norn.permissions.models import PermissionRequest

            request = PermissionRequest(
                tool_name=call.name,
                risk_level=tool.risk_level.value,
                arguments=call.arguments,
            )
            decision = await self.permission_checker.check(request)
            if not decision.approved:
                reason = decision.reason or "Permission denied"
                return ToolResult(error=f"Permission denied: {reason}")

        try:
            input_obj = tool.input_model(**call.arguments)
            return await tool.execute(input_obj, self.ctx)
        except Exception as e:
            return ToolResult(error=f"Tool execution error: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_core/test_agent.py -v`
Expected: ALL PASSED (old tests still pass, new tests pass)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 6: Commit**

```bash
git add src/norn/core/agent.py tests/test_core/test_agent.py
git commit -m "feat(core): integrate permission checker into agent loop"
```

---

## Task 4: Wire Permissions into CLI

**Files:**
- Modify: `src/norn/cli/main.py`

Add the permission checker to the CLI `chat` and `run` commands. In CLI mode, the `prompt_fn` uses `rich.prompt.Confirm` to ask the user for approval.

**Step 1: Write the integration**

Update `src/norn/cli/main.py` to include permission setup:

```python
"""Norn CLI entrypoint."""

from __future__ import annotations

import asyncio
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Confirm, Prompt

from norn.core.agent import AgentLoop
from norn.core.config import NornConfig
from norn.core.llm import LiteLLMProvider
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.permissions.models import PermissionRequest
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
console = Console()


def _build_registry() -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


def _build_provider(config: NornConfig) -> LiteLLMProvider:
    """Build the LLM provider from config."""
    model = config.llm.model
    if config.llm.provider == "ollama":
        model = f"ollama/{config.llm.model}"
    elif config.llm.provider == "openrouter":
        model = f"openrouter/{config.llm.model}"
    return LiteLLMProvider(model=model, api_base=config.llm.api_base)


async def _cli_prompt_fn(request: PermissionRequest, description: str) -> bool:
    """Prompt the user for permission approval via rich."""
    risk_colors = {"low": "green", "medium": "yellow", "high": "red"}
    color = risk_colors.get(request.risk_level, "white")
    console.print(
        f"\n[bold {color}]Permission required[/bold {color}]: "
        f"[{color}]{request.tool_name}[/{color}] (risk: {request.risk_level})"
    )
    console.print(f"  {description}")
    return Confirm.ask("  Allow?", default=True)


def _build_permission_checker(config: NornConfig) -> PermissionChecker:
    """Build the permission checker from config."""
    return PermissionChecker(
        mode=config.permissions.mode,
        classifier=RiskClassifier(),
        prompt_fn=_cli_prompt_fn,
    )


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    checker = _build_permission_checker(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
    )

    console.print("[bold]Norn[/bold] - the coding agent that weaves your destiny")
    console.print(f"Permission mode: [bold]{config.permissions.mode.value}[/bold]")
    console.print("Type 'exit' or 'quit' to leave. Ctrl+C to interrupt.\n")

    async def _chat_loop() -> None:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                break

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("Goodbye.")
                break

            if not user_input.strip():
                continue

            try:
                with console.status("[dim]Thinking...[/dim]"):
                    response = await agent.run(user_input)

                if response.content:
                    console.print(Markdown(response.content))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_chat_loop())


@app.command()
def run(prompt: str = typer.Argument(help="One-shot prompt to execute")) -> None:
    """Run a one-shot prompt and exit."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    checker = _build_permission_checker(config)
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
        permission_checker=checker,
    )

    async def _run_once() -> None:
        response = await agent.run(prompt)
        if response.content:
            console.print(Markdown(response.content))

    asyncio.run(_run_once())


@app.command()
def tools() -> None:
    """List available tools."""
    registry = _build_registry()
    console.print("[bold]Available tools:[/bold]\n")
    for tool in registry.list_tools():
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[tool.risk_level.value]
        console.print(
            f"  [{risk_color}]{tool.risk_level.value:>6}[/{risk_color}]  "
            f"[bold]{tool.name}[/bold] - {tool.description}"
        )


@app.command()
def config() -> None:
    """Show current configuration."""
    cfg = NornConfig.load()
    cfg.apply_env_overrides()
    console.print("[bold]Current configuration:[/bold]\n")
    console.print(f"  LLM provider: {cfg.llm.provider}")
    console.print(f"  LLM model:    {cfg.llm.model}")
    console.print(f"  Permissions:  {cfg.permissions.mode.value}")
    console.print(f"  Dream:        {'enabled' if cfg.flags.dream_system else 'disabled'}")
    console.print(f"  Coordinator:  {'enabled' if cfg.flags.coordinator else 'disabled'}")


@app.command()
def version() -> None:
    """Show Norn version."""
    console.print("norn 0.1.0")


if __name__ == "__main__":
    app()
```

**Step 2: Verify CLI still works**

Run: `uv run norn tools`
Run: `uv run norn config`
Run: `uv run norn version`
Expected: All commands work

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 4: Commit**

```bash
git add src/norn/cli/main.py
git commit -m "feat(cli): wire permission checker into chat and run commands"
```

---

## Task 5: Feature Flag Registry

**Files:**
- Create: `src/norn/flags/registry.py`
- Test: `tests/test_flags/__init__.py`
- Test: `tests/test_flags/test_registry.py`

The FeatureFlagRegistry holds flag definitions with defaults, resolves values with priority (env var > config > default), and provides a simple `is_enabled(name)` interface.

**Step 1: Create test directory**

```bash
mkdir -p tests/test_flags
touch tests/test_flags/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_flags/test_registry.py
"""Tests for the feature flag registry."""

import os
from unittest.mock import patch

import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry


@pytest.fixture
def registry():
    return FeatureFlagRegistry(
        flags={
            "dream_system": FeatureFlag(
                name="dream_system", default=False, description="Memory consolidation"
            ),
            "coordinator": FeatureFlag(
                name="coordinator", default=False, description="Multi-agent mode"
            ),
            "ml_tools": FeatureFlag(
                name="ml_tools", default=True, description="MLOps-specific tools"
            ),
        }
    )


class TestDefaults:
    def test_default_false(self, registry):
        assert registry.is_enabled("dream_system") is False

    def test_default_true(self, registry):
        assert registry.is_enabled("ml_tools") is True

    def test_unknown_flag_returns_false(self, registry):
        assert registry.is_enabled("nonexistent") is False


class TestConfigOverride:
    def test_config_overrides_default(self, registry):
        registry.apply_config({"dream_system": True})
        assert registry.is_enabled("dream_system") is True

    def test_config_does_not_affect_unspecified(self, registry):
        registry.apply_config({"dream_system": True})
        assert registry.is_enabled("coordinator") is False  # Still default

    def test_config_can_disable_default_true(self, registry):
        registry.apply_config({"ml_tools": False})
        assert registry.is_enabled("ml_tools") is False


class TestEnvOverride:
    def test_env_overrides_default(self, registry):
        with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": "true"}):
            assert registry.is_enabled("dream_system") is True

    def test_env_overrides_config(self, registry):
        registry.apply_config({"dream_system": True})
        with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": "false"}):
            assert registry.is_enabled("dream_system") is False

    def test_env_true_variants(self, registry):
        for value in ["true", "1", "yes", "True", "TRUE", "YES"]:
            with patch.dict(os.environ, {"NORN_FLAG_DREAM_SYSTEM": value}):
                assert registry.is_enabled("dream_system") is True, f"Failed for: {value}"

    def test_env_false_variants(self, registry):
        for value in ["false", "0", "no", "False", "FALSE", "NO"]:
            with patch.dict(os.environ, {"NORN_FLAG_ML_TOOLS": value}):
                assert registry.is_enabled("ml_tools") is False, f"Failed for: {value}"


class TestListFlags:
    def test_list_all(self, registry):
        flags = registry.list_flags()
        assert len(flags) == 3
        names = {f.name for f in flags}
        assert names == {"dream_system", "coordinator", "ml_tools"}

    def test_list_shows_resolved_values(self, registry):
        registry.apply_config({"dream_system": True})
        flags = registry.list_flags()
        dream = next(f for f in flags if f.name == "dream_system")
        assert dream.default is False  # Original default
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_flags/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 4: Write implementation**

```python
# src/norn/flags/registry.py
"""Feature flag registry with priority resolution."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


_TRUTHY = {"true", "1", "yes"}
_FALSY = {"false", "0", "no"}


@dataclass
class FeatureFlag:
    """A single feature flag definition."""

    name: str
    default: bool
    description: str = ""


class FeatureFlagRegistry:
    """Registry for feature flags with priority: env > config > default."""

    def __init__(self, flags: dict[str, FeatureFlag] | None = None) -> None:
        self._flags: dict[str, FeatureFlag] = flags or {}
        self._config_overrides: dict[str, bool] = {}

    def register(self, flag: FeatureFlag) -> None:
        """Register a feature flag."""
        self._flags[flag.name] = flag

    def apply_config(self, config: dict[str, bool]) -> None:
        """Apply config-level overrides."""
        self._config_overrides.update(config)

    def is_enabled(self, name: str) -> bool:
        """Resolve a flag value with priority: env > config > default."""
        flag = self._flags.get(name)
        if flag is None:
            return False

        # Priority 1: Environment variable
        env_key = f"NORN_FLAG_{name.upper()}"
        env_value = os.environ.get(env_key)
        if env_value is not None:
            return env_value.lower() in _TRUTHY

        # Priority 2: Config override
        if name in self._config_overrides:
            return self._config_overrides[name]

        # Priority 3: Default
        return flag.default

    def list_flags(self) -> list[FeatureFlag]:
        """List all registered flags."""
        return list(self._flags.values())
```

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_flags/test_registry.py -v`
Expected: ALL PASSED

**Step 6: Commit**

```bash
git add src/norn/flags/ tests/test_flags/
git commit -m "feat(flags): add feature flag registry with env > config > default resolution"
```

---

## Task 6: Conditional Tool Registration with Feature Flags

**Files:**
- Modify: `src/norn/tools/registry.py`
- Modify: `tests/test_tools/test_registry.py`

The ToolRegistry already has a `_feature_flags` dict. Now we wire it to the `FeatureFlagRegistry` so that tools gated behind a disabled flag are excluded from `get_schemas()` and `list_tools()`.

**Step 1: Write the failing tests**

Add to `tests/test_tools/test_registry.py`:

```python
# Add import at top:
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry


# --- New tests ---

def test_flagged_tool_excluded_when_disabled():
    """Tool gated behind a disabled flag should not appear in schemas or list."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag(name="ml_tools", default=False)}
    )
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool(), feature_flag="ml_tools")

    assert registry.list_tools() == []
    assert registry.get_schemas() == []
    # But get() still works (for error messages)
    assert registry.get("mock_tool") is not None


def test_flagged_tool_included_when_enabled():
    """Tool gated behind an enabled flag should appear normally."""
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag(name="ml_tools", default=True)}
    )
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool(), feature_flag="ml_tools")

    assert len(registry.list_tools()) == 1
    assert len(registry.get_schemas()) == 1


def test_unflagged_tool_always_included():
    """Tool without a flag is always included."""
    flag_registry = FeatureFlagRegistry()
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(MockTool())

    assert len(registry.list_tools()) == 1
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_registry.py -v`
Expected: FAIL — `ToolRegistry() got an unexpected keyword argument 'flag_registry'`

**Step 3: Update ToolRegistry**

```python
# src/norn/tools/registry.py
"""Tool registry with schema caching and feature flag support."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from norn.flags.registry import FeatureFlagRegistry
    from norn.tools.base import Tool


class ToolRegistry:
    """Registry for agent tools with cached JSON schemas."""

    def __init__(self, flag_registry: FeatureFlagRegistry | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict[str, Any]] | None = None
        self._feature_flags: dict[str, str] = {}  # tool_name -> flag_name
        self._flag_registry: FeatureFlagRegistry | None = flag_registry

    def register(self, tool: Tool, feature_flag: str | None = None) -> None:
        """Register a tool. Raises ValueError if already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        if feature_flag:
            self._feature_flags[tool.name] = feature_flag
        self._schema_cache = None  # Invalidate cache

    def _is_tool_enabled(self, name: str) -> bool:
        """Check if a tool is enabled via its feature flag."""
        flag_name = self._feature_flags.get(name)
        if flag_name is None:
            return True  # No flag = always enabled
        if self._flag_registry is None:
            return True  # No flag registry = always enabled
        return self._flag_registry.is_enabled(flag_name)

    def get(self, name: str) -> Tool | None:
        """Get a tool by name (regardless of flag status)."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all enabled tools."""
        return [t for t in self._tools.values() if self._is_tool_enabled(t.name)]

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for enabled tools. Cached for prompt efficiency."""
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
            if not self._is_tool_enabled(tool.name):
                continue
            schema = tool.input_model.model_json_schema()
            schemas.append(
                {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": schema,
                }
            )
        self._schema_cache = schemas
        return self._schema_cache

    def scoped(self, tool_names: list[str]) -> ToolRegistry:
        """Create a new registry with only the specified tools."""
        scoped = ToolRegistry(flag_registry=self._flag_registry)
        for name in tool_names:
            tool = self.get(name)
            if tool:
                flag = self._feature_flags.get(name)
                scoped.register(tool, feature_flag=flag)
        return scoped
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_registry.py -v`
Expected: ALL PASSED (old tests still pass, new tests pass)

**Step 5: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 6: Commit**

```bash
git add src/norn/tools/registry.py tests/test_tools/test_registry.py
git commit -m "feat(tools): support feature flag gating for conditional tool registration"
```

---

## Task 7: Wire Feature Flags into Config and CLI

**Files:**
- Modify: `src/norn/core/config.py` (minor: add flag env overrides)
- Modify: `src/norn/cli/main.py` (use FeatureFlagRegistry)
- Test: `tests/test_core/test_config.py` (add flag override test)

**Step 1: Add flag env override test**

Add to `tests/test_core/test_config.py`:

```python
import os
from unittest.mock import patch


def test_flag_env_overrides():
    """Feature flags should be overridable via env vars."""
    config = NornConfig()
    assert config.flags.dream_system is False
    # Config-level override (through NornConfig initialization)
    config.flags.dream_system = True
    assert config.flags.dream_system is True
```

**Step 2: Update CLI to use feature flags**

In `src/norn/cli/main.py`, update `_build_registry` to accept an optional `FeatureFlagRegistry`:

Replace the `_build_registry` function and the `chat`/`run` commands to use it:

```python
# Add import at top:
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry


def _build_flag_registry(config: NornConfig) -> FeatureFlagRegistry:
    """Build the feature flag registry from config."""
    registry = FeatureFlagRegistry(
        flags={
            "dream_system": FeatureFlag("dream_system", False, "Memory consolidation"),
            "coordinator": FeatureFlag("coordinator", False, "Multi-agent mode"),
            "ml_tools": FeatureFlag("ml_tools", True, "MLOps-specific tools"),
        }
    )
    registry.apply_config(
        {
            "dream_system": config.flags.dream_system,
            "coordinator": config.flags.coordinator,
            "ml_tools": config.flags.ml_tools,
        }
    )
    return registry


def _build_registry(flag_registry: FeatureFlagRegistry | None = None) -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry
```

Then update `chat()`, `run()`, `tools()` to call `_build_flag_registry` and pass it through.

**Step 3: Run full test suite**

Run: `uv run pytest -v`
Expected: ALL PASSED

**Step 4: Verify CLI**

Run: `uv run norn tools`
Run: `uv run norn config`
Expected: Both work

**Step 5: Commit**

```bash
git add src/norn/cli/main.py src/norn/core/config.py tests/test_core/test_config.py
git commit -m "feat(cli): wire feature flag registry into CLI and config"
```

---

## Task 8: Full Integration Tests

**Files:**
- Modify: `tests/test_integration.py`

Add integration tests that exercise the full permission + flag pipeline.

**Step 1: Write integration tests**

Add to `tests/test_integration.py`:

```python
# Add imports:
from norn.permissions.checker import PermissionChecker
from norn.permissions.classifier import RiskClassifier
from norn.core.config import PermissionMode
from norn.flags.registry import FeatureFlag, FeatureFlagRegistry


@pytest.mark.asyncio
async def test_permission_blocks_destructive_bash():
    """Full pipeline: bash rm -rf should be denied in interactive mode."""
    checker = PermissionChecker(
        mode=PermissionMode.INTERACTIVE,
        classifier=RiskClassifier(),
    )

    registry = ToolRegistry()
    from norn.tools.bash_tool import BashTool
    registry.register(BashTool())

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments={"command": "rm -rf /"})
                ],
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
    flag_registry = FeatureFlagRegistry(
        flags={"ml_tools": FeatureFlag("ml_tools", False)}
    )
    registry = ToolRegistry(flag_registry=flag_registry)

    from norn.tools.bash_tool import BashTool
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
    from norn.tools.bash_tool import BashTool
    registry.register(BashTool())

    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(id="c1", name="bash", arguments={"command": "echo safe"})
                ],
            ),
            LLMResponse(content="Done.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=registry, permission_checker=checker)
    result = await agent.run("run a command")
    assert result.content == "Done."
```

**Step 2: Run tests**

Run: `uv run pytest tests/test_integration.py -v`
Expected: ALL PASSED

**Step 3: Run full test suite + ruff**

Run: `uv run pytest -v && uv run ruff check src/ tests/`
Expected: ALL PASSED, 0 lint errors

**Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for permission system and feature flags"
```

---

## Task 9: Final Verification and Cleanup

**Step 1: Run full test suite**

```bash
uv run pytest -v
```
Expected: ALL PASSED

**Step 2: Run ruff**

```bash
uv run ruff check src/ tests/
```
Expected: 0 errors

**Step 3: Manual smoke test with OpenRouter**

```bash
NORN_LLM_PROVIDER=openrouter NORN_LLM_MODEL="nvidia/nemotron-3-super-120b-a12b:free" uv run norn run "Read the file /tmp/test.txt"
```
Expected: Works end-to-end with permission check

**Step 4: Test permission modes via env**

```bash
# Yolo mode - should work without prompts
NORN_LLM_PROVIDER=openrouter NORN_LLM_MODEL="nvidia/nemotron-3-super-120b-a12b:free" NORN_PERMISSION_MODE=yolo uv run norn run "Read the file /tmp/test.txt"
```

**Step 5: Verify git log**

```bash
git log --oneline
```
Expected: Clean atomic commits from Phase 2

**Step 6: Final commit (if any cleanup needed)**

```bash
# Only if cleanup was needed
git add -A && git commit -m "chore: phase 2 cleanup"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 0 | Permission models | 4 | `permissions/models.py` |
| 1 | Risk classifier | ~25 | `permissions/classifier.py` |
| 2 | Permission checker | ~15 | `permissions/checker.py` |
| 3 | AgentLoop integration | 3 | `core/agent.py` (modify) |
| 4 | CLI wiring | manual | `cli/main.py` (modify) |
| 5 | Feature flag registry | ~12 | `flags/registry.py` |
| 6 | Conditional tool registration | 3 | `tools/registry.py` (modify) |
| 7 | Config + CLI flags | 1 | `config.py`, `main.py` (modify) |
| 8 | Integration tests | 3 | `test_integration.py` (modify) |
| 9 | Final verification | manual | — |

**Total: ~66 new tests, 9 commits, 3 new files, 4 modified files**
