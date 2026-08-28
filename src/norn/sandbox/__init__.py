"""Fail-closed execution sandbox (SOTA v2, workstream C).

Confinement is applied by argv-wrapping the shell command with a platform
mechanism (macOS seatbelt today; Linux bubblewrap/Landlock is a documented
future extension). The invariant is fail-closed: when confinement is
requested but the mechanism is unavailable, execution is REFUSED — never
silently run unconfined.
"""

from norn.sandbox.models import SandboxPolicy
from norn.sandbox.seatbelt import build_argv, is_available

__all__ = ["SandboxPolicy", "build_argv", "is_available"]
