"""Sandbox policy model."""

from __future__ import annotations

from enum import StrEnum


class SandboxPolicy(StrEnum):
    """Per-call confinement policy for sandboxed command execution.

    - READ_ONLY: filesystem reads only (plus /dev/null|stdout|stderr).
    - WORKSPACE_WRITE: reads everywhere, writes confined to the workspace
      and temp directories. The sensible default.
    - DANGER_FULL_ACCESS: no confinement at all — only reachable through an
      approved escalation (user prompt), never silently.
    """

    READ_ONLY = "read-only"
    WORKSPACE_WRITE = "workspace-write"
    DANGER_FULL_ACCESS = "danger-full-access"
