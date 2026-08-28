"""Tests for the fail-closed bash sandbox (SOTA v2, workstream C).

Platform-agnostic tests mock the mechanism; real seatbelt integration tests
are skipif-gated to macOS (first skipif in the suite — CI runs Linux).
"""

from __future__ import annotations

import sys
from unittest.mock import AsyncMock

import pytest

from norn.sandbox import SandboxPolicy, build_argv
from norn.sandbox.seatbelt import SANDBOX_EXEC, build_profile
from norn.tools.base import ToolContext
from norn.tools.bash_tool import BashInput, BashTool

# --------------------------------------------------------------------------- #
# build_argv / build_profile structure (no platform dependency)
# --------------------------------------------------------------------------- #


def test_build_argv_structure(tmp_path):
    argv = build_argv(SandboxPolicy.WORKSPACE_WRITE, str(tmp_path), "echo hi")
    assert argv[0] == SANDBOX_EXEC
    assert argv[1] == "-p"
    assert "(deny default)" in argv[2]
    assert argv[3] == "-D"
    assert argv[4].startswith("WORKSPACE_DIR=")
    assert argv[5] == "-D"
    assert argv[6].startswith("TEMP_DIR=")
    assert argv[-3:] == ["/bin/sh", "-c", "echo hi"]


def test_workspace_write_profile_allows_workspace_and_tmp():
    profile = build_profile(
        SandboxPolicy.WORKSPACE_WRITE, allow_network=False, extra_write_paths=[]
    )
    assert '(allow file-write* (subpath (param "WORKSPACE_DIR")))' in profile
    assert '(subpath "/private/tmp")' in profile
    assert "(allow network*)" not in profile


def test_read_only_profile_has_no_workspace_write():
    profile = build_profile(SandboxPolicy.READ_ONLY, allow_network=False, extra_write_paths=[])
    assert "WORKSPACE_DIR" not in profile
    assert '(allow file-write* (subpath' not in profile
    # stdout/stderr writes stay allowed
    assert '"/dev/null"' in profile


def test_network_opt_in():
    profile = build_profile(SandboxPolicy.WORKSPACE_WRITE, allow_network=True, extra_write_paths=[])
    assert "(allow network*)" in profile


def test_extra_write_paths_injected(tmp_path):
    profile = build_profile(
        SandboxPolicy.WORKSPACE_WRITE, allow_network=False, extra_write_paths=[str(tmp_path)]
    )
    assert str(tmp_path.resolve()) in profile


def test_build_argv_rejects_danger_full_access(tmp_path):
    with pytest.raises(ValueError, match="unconfined"):
        build_argv(SandboxPolicy.DANGER_FULL_ACCESS, str(tmp_path), "echo hi")


# --------------------------------------------------------------------------- #
# BashTool behaviour with the mechanism mocked
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sandbox_off_runs_unconfined(tmp_path):
    tool = BashTool()  # defaults: sandbox disabled
    result = await tool.execute(BashInput(command="echo plain"), ToolContext(cwd=str(tmp_path)))
    assert result.output is not None and "plain" in result.output


@pytest.mark.asyncio
async def test_fail_closed_when_mechanism_unavailable(tmp_path, monkeypatch):
    """Confinement requested + unavailable → refusal, and NO subprocess spawned."""
    import norn.sandbox as sandbox_pkg
    import norn.tools.bash_tool as bash_mod

    monkeypatch.setattr(sandbox_pkg, "is_available", lambda: False)
    spawn_shell = AsyncMock(side_effect=AssertionError("must not spawn"))
    spawn_exec = AsyncMock(side_effect=AssertionError("must not spawn"))
    monkeypatch.setattr(bash_mod.asyncio, "create_subprocess_shell", spawn_shell)
    monkeypatch.setattr(bash_mod.asyncio, "create_subprocess_exec", spawn_exec)

    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(BashInput(command="echo hi"), ToolContext(cwd=str(tmp_path)))
    assert result.is_error
    assert result.error_type == "SandboxUnavailable"
    spawn_shell.assert_not_awaited()
    spawn_exec.assert_not_awaited()


@pytest.mark.asyncio
async def test_invalid_policy_is_refused(tmp_path):
    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(
        BashInput(command="echo hi", sandbox_policy="yolo-mode"),
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.is_error
    assert result.error_type == "InvalidArgument"


@pytest.mark.asyncio
async def test_escalation_denied_without_handler(tmp_path):
    tool = BashTool(sandbox_enabled=True, escalation_fn=None)
    result = await tool.execute(
        BashInput(command="echo hi", sandbox_policy="danger-full-access"),
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.is_error
    assert result.error_type == "PermissionDenied"
    assert "no escalation handler" in result.error


@pytest.mark.asyncio
async def test_escalation_refused_by_user(tmp_path):
    async def deny(request, description):
        return False

    tool = BashTool(sandbox_enabled=True, escalation_fn=deny)
    result = await tool.execute(
        BashInput(command="echo hi", sandbox_policy="danger-full-access"),
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.is_error
    assert result.error_type == "PermissionDenied"


@pytest.mark.asyncio
async def test_escalation_approved_runs_unconfined(tmp_path):
    prompts: list = []

    async def approve(request, description):
        prompts.append((request, description))
        return True

    tool = BashTool(sandbox_enabled=True, escalation_fn=approve)
    result = await tool.execute(
        BashInput(command="echo escalated", sandbox_policy="danger-full-access"),
        ToolContext(cwd=str(tmp_path)),
    )
    assert not result.is_error
    assert "escalated" in result.output
    assert len(prompts) == 1
    assert prompts[0][0].arguments["sandbox_policy"] == "danger-full-access"


@pytest.mark.asyncio
async def test_confined_call_uses_sandbox_argv(tmp_path, monkeypatch):
    """When available, the subprocess is spawned through sandbox-exec argv."""
    import norn.sandbox as sandbox_pkg
    import norn.tools.bash_tool as bash_mod

    monkeypatch.setattr(sandbox_pkg, "is_available", lambda: True)
    seen: dict = {}

    async def spy_exec(*argv, **kwargs):
        seen["argv"] = list(argv)
        # Substitute a plain shell run so the test passes on any platform
        return await bash_mod.asyncio.create_subprocess_shell(argv[-1], **kwargs)

    monkeypatch.setattr(bash_mod.asyncio, "create_subprocess_exec", spy_exec)

    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(BashInput(command="echo confined"), ToolContext(cwd=str(tmp_path)))
    assert not result.is_error
    assert seen["argv"][0] == SANDBOX_EXEC
    assert "(deny default)" in seen["argv"][2]


# --------------------------------------------------------------------------- #
# Real seatbelt integration (macOS only)
# --------------------------------------------------------------------------- #

darwin_only = pytest.mark.skipif(sys.platform != "darwin", reason="seatbelt requires macOS")


@darwin_only
@pytest.mark.asyncio
async def test_seatbelt_workspace_write_allows_workspace(tmp_path):
    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(
        BashInput(command="echo data > f.txt && cat f.txt"), ToolContext(cwd=str(tmp_path))
    )
    assert not result.is_error, result.error
    assert "data" in result.output
    assert (tmp_path / "f.txt").exists()


@darwin_only
@pytest.mark.asyncio
async def test_seatbelt_denies_write_outside_workspace(tmp_path):
    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(
        BashInput(command="touch /usr/local/__norn_sandbox_test__"),
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.is_error
    assert "Exit code" in result.error


@darwin_only
@pytest.mark.asyncio
async def test_seatbelt_read_only_denies_workspace_write(tmp_path):
    tool = BashTool(sandbox_enabled=True, default_policy="read-only")
    write = await tool.execute(
        BashInput(command="echo x > f.txt"), ToolContext(cwd=str(tmp_path))
    )
    assert write.is_error
    read = await tool.execute(BashInput(command="ls"), ToolContext(cwd=str(tmp_path)))
    assert not read.is_error


@darwin_only
@pytest.mark.asyncio
async def test_seatbelt_denies_network_by_default(tmp_path):
    tool = BashTool(sandbox_enabled=True)
    result = await tool.execute(
        BashInput(command="curl -m 3 -sS https://example.com", timeout=20),
        ToolContext(cwd=str(tmp_path)),
    )
    assert result.is_error
