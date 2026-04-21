"""Tests for tools.base — covers ToolErrorType taxonomy."""

from __future__ import annotations

from norn.tools.base import ToolErrorType


def test_tool_error_type_values() -> None:
    """The 10 documented categories must keep their stable string values
    (these end up in JSONL logs and dashboards key on them)."""
    assert ToolErrorType.FILE_NOT_FOUND.value == "FileNotFound"
    assert ToolErrorType.PERMISSION_DENIED.value == "PermissionDenied"
    assert ToolErrorType.INVALID_ARGUMENT.value == "InvalidArgument"
    assert ToolErrorType.TIMEOUT.value == "Timeout"
    assert ToolErrorType.NETWORK_ERROR.value == "NetworkError"
    assert ToolErrorType.HTTP_ERROR.value == "HttpError"
    assert ToolErrorType.PARSE_ERROR.value == "ParseError"
    assert ToolErrorType.NOT_SUPPORTED.value == "NotSupported"
    assert ToolErrorType.EXECUTION_ERROR.value == "ExecutionError"
    assert ToolErrorType.RESOURCE_EXHAUSTED.value == "ResourceExhausted"


def test_tool_error_type_is_str_enum() -> None:
    """StrEnum members ARE strings — callers can pass the member directly
    into ``ToolResult(error_type=...)`` without ``.value`` boilerplate."""
    assert ToolErrorType.FILE_NOT_FOUND == "FileNotFound"
    assert isinstance(ToolErrorType.FILE_NOT_FOUND, str)
