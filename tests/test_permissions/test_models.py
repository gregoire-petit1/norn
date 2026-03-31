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
