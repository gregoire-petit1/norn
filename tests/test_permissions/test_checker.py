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
        req = PermissionRequest(tool_name="bash", risk_level="high", arguments={"command": "ls"})
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
        req = PermissionRequest(tool_name="bash", risk_level="high", arguments={"command": "ls"})
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
