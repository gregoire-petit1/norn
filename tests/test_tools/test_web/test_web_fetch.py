"""Tests for WebFetchTool."""

import httpx
import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.web.web_fetch import WebFetchInput, WebFetchTool


@pytest.fixture
def tool():
    return WebFetchTool()


def test_tool_metadata(tool):
    assert tool.name == "web_fetch"
    assert tool.risk_level == RiskLevel.LOW
    assert tool.description


@pytest.mark.asyncio
async def test_fetch_plain_text(tool, tmp_path):
    """Fetch a plain text response."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="Hello from the web!")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://example.com/text.txt"), ctx, _transport=transport
    )
    assert result.is_error is False
    assert "Hello from the web!" in result.output


@pytest.mark.asyncio
async def test_fetch_html_strips_tags(tool, tmp_path):
    """HTML responses should have tags stripped."""
    html = "<html><head><title>My Page</title></head><body><h1>Hello</h1><p>World</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=html)

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://example.com/"), ctx, _transport=transport
    )
    assert result.is_error is False
    assert "<html>" not in result.output
    assert "Hello" in result.output
    assert "World" in result.output


@pytest.mark.asyncio
async def test_fetch_respects_max_chars(tool, tmp_path):
    """Output should be truncated at max_chars."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="A" * 50000)

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://example.com/", max_chars=100), ctx, _transport=transport
    )
    assert result.is_error is False
    assert len(result.output) <= 150  # slight buffer for truncation message


@pytest.mark.asyncio
async def test_fetch_http_error(tool, tmp_path):
    """HTTP 404 should return an error."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, text="Not Found")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://example.com/missing"), ctx, _transport=transport
    )
    assert result.is_error is True
    assert "404" in result.error


@pytest.mark.asyncio
async def test_fetch_connection_error(tool, tmp_path):
    """Connection errors should return an error result (not raise)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://unreachable.example.com/"), ctx, _transport=transport
    )
    assert result.is_error is True


# --------------------------------------------------------------------------- #
# B1.5 - ToolErrorType taxonomy
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_fetch_http_error_sets_http_error_type(tool, tmp_path):
    """HTTP 4xx/5xx tags as HTTP_ERROR."""
    from norn.tools.base import ToolErrorType

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Server Error")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://example.com/boom"), ctx, _transport=transport
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.HTTP_ERROR


@pytest.mark.asyncio
async def test_fetch_connection_error_sets_network_error_type(tool, tmp_path):
    """ConnectError / TimeoutException tag as NETWORK_ERROR."""
    from norn.tools.base import ToolErrorType

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebFetchInput(url="https://unreachable.example.com/"), ctx, _transport=transport
    )
    assert result.is_error is True
    assert result.error_type == ToolErrorType.NETWORK_ERROR


@pytest.mark.asyncio
async def test_fetch_missing_httpx_sets_not_supported(tool, tmp_path, monkeypatch):
    """Missing httpx tags as NOT_SUPPORTED."""
    import builtins

    from norn.tools.base import ToolErrorType

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "httpx":
            raise ImportError("no httpx")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(WebFetchInput(url="https://example.com/"), ctx)
    assert result.is_error is True
    assert result.error_type == ToolErrorType.NOT_SUPPORTED
