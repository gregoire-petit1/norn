"""Permission checker: enforces permission modes before tool execution."""

from __future__ import annotations

import contextlib
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from norn.core.config import PermissionMode
    from norn.permissions.classifier import RiskClassifier

from norn.observability import EventName, get_logger
from norn.permissions.models import PermissionDecision, PermissionRequest
from norn.tools.base import RiskLevel

# Type alias for the user-prompt callback
PromptFn = Callable[[PermissionRequest, str], Awaitable[bool]]

# Risk level ordering for comparison
_RISK_ORDER = {RiskLevel.LOW.value: 0, RiskLevel.MEDIUM.value: 1, RiskLevel.HIGH.value: 2}

_log = get_logger(__name__)


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
        decision = await self._apply_mode(request, effective_risk, escalation.is_destructive)

        # Step 3: One-shot observability event. Event field is named
        # ``granted`` per the design doc (see
        # ``docs/plans/2026-04-21-norn-phase8-observability-design.md``
        # §permission.decision) even though the Python attribute on
        # :class:`PermissionDecision` is ``approved``. Fail-open: a logging
        # failure must never affect permission control flow.
        with contextlib.suppress(Exception):
            _log.info(
                EventName.PERMISSION_DECISION,
                tool_name=request.tool_name,
                mode=self.mode.value,
                risk_level=effective_risk,
                granted=decision.approved,
                reason=decision.reason,
            )
        return decision

    async def _apply_mode(
        self,
        request: PermissionRequest,
        effective_risk: str,
        is_destructive: bool,
    ) -> PermissionDecision:
        """Apply the permission mode rules."""
        mode = self.mode.value

        if mode == "yolo":
            return PermissionDecision(approved=True, reason="yolo")

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
                return PermissionDecision(approved=True, reason="auto_approved_low_medium")
            return await self._prompt_or_deny(request, effective_risk)

        # interactive (default)
        # LOW auto-approved, MEDIUM + HIGH need prompt
        if _RISK_ORDER.get(effective_risk, 2) <= _RISK_ORDER["low"]:
            return PermissionDecision(approved=True, reason="auto_approved_low")
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
            reason="user_approved" if approved else "user_denied",
        )
