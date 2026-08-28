"""macOS seatbelt confinement via /usr/bin/sandbox-exec.

sandbox-exec is deprecated by Apple but functional; if it ever disappears,
:func:`is_available` returns False and callers refuse to execute (fail-closed)
rather than falling back to unconfined execution.

Profile design: deny-by-default, global reads allowed (dyld, toolchains —
read control of secrets stays with the permissions RiskClassifier), writes
confined to the workspace + temp dirs, network opt-in. Paths are passed via
``-D`` parameters and resolved with realpath first: macOS symlinks
(/tmp → /private/tmp, /var → /private/var) would otherwise defeat
``subpath`` matching.

Linux (bubblewrap/Landlock) is a planned extension with the same contract.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

from norn.sandbox.models import SandboxPolicy

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# Common profile prefix: deny everything, then re-allow what any normal
# CLI process needs to start and read. mach-lookup is deliberately broad —
# without it most macOS binaries crash on launch.
_PROFILE_BASE = """\
(version 1)
(deny default)
(allow process-fork)
(allow process-exec*)
(allow file-read*)
(allow file-read-metadata)
(allow sysctl-read)
(allow mach-lookup)
(allow signal)
(allow file-ioctl (path "/dev/dtracehelper"))
(allow file-write-data (path "/dev/null") (path "/dev/stdout") (path "/dev/stderr"))
"""

_WRITE_PARAMS = """\
(allow file-write* (subpath (param "WORKSPACE_DIR")))
(allow file-write* (subpath "/private/tmp"))
(allow file-write* (subpath (param "TEMP_DIR")))
"""

_NETWORK = "(allow network*)\n"


def is_available() -> bool:
    """True when seatbelt confinement can actually be applied here."""
    return sys.platform == "darwin" and os.access(SANDBOX_EXEC, os.X_OK)


def _escape_path(path: str) -> str:
    """Escape a filesystem path for embedding in a seatbelt string literal."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def build_profile(
    policy: SandboxPolicy, *, allow_network: bool, extra_write_paths: list[str]
) -> str:
    """Assemble the seatbelt profile text for a policy."""
    profile = _PROFILE_BASE
    if policy == SandboxPolicy.WORKSPACE_WRITE:
        profile += _WRITE_PARAMS
        for extra in extra_write_paths:
            resolved = _escape_path(str(Path(extra).expanduser().resolve()))
            profile += f'(allow file-write* (subpath "{resolved}"))\n'
    if allow_network:
        profile += _NETWORK
    return profile


def build_argv(
    policy: SandboxPolicy,
    workspace: str,
    script: str,
    *,
    allow_network: bool = False,
    extra_write_paths: list[str] | None = None,
) -> list[str]:
    """Full argv for a confined `/bin/sh -c <script>` under sandbox-exec.

    The wrapped script (markers, rc capture) runs unchanged INSIDE the
    sandbox, so callers keep identical shell semantics. DANGER_FULL_ACCESS
    is not a valid input here — unconfined execution never goes through
    the sandbox layer.
    """
    if policy == SandboxPolicy.DANGER_FULL_ACCESS:
        raise ValueError("danger-full-access is unconfined; do not wrap it")
    profile = build_profile(
        policy, allow_network=allow_network, extra_write_paths=extra_write_paths or []
    )
    return [
        SANDBOX_EXEC,
        "-p",
        profile,
        "-D",
        f"WORKSPACE_DIR={Path(workspace).resolve()}",
        "-D",
        f"TEMP_DIR={Path(tempfile.gettempdir()).resolve()}",
        "/bin/sh",
        "-c",
        script,
    ]
