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
