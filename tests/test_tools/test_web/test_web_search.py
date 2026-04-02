"""Tests for WebSearchTool."""

import httpx
import pytest

from norn.tools.base import RiskLevel, ToolContext
from norn.tools.web.web_search import WebSearchInput, WebSearchTool

# Minimal DuckDuckGo Lite-style HTML fixture for testing
_DDG_HTML = """\
<html><body>
<form method="post" action="/lite/">
  <input type="hidden" name="kl" value="us-en"/>
</form>
<table>
  <tr>
    <td class="result-link">
      <a href="https://python.org/asyncio">Python asyncio — Official Docs</a>
    </td>
  </tr>
  <tr>
    <td class="result-snippet">
      asyncio is a library to write concurrent code using the async/await syntax.
    </td>
  </tr>
  <tr>
    <td class="result-link">
      <a href="https://realpython.com/async-io">Async IO in Python: A Complete Walkthrough</a>
    </td>
  </tr>
  <tr>
    <td class="result-snippet">
      A complete tutorial on async I/O in Python including examples and best practices.
    </td>
  </tr>
</table>
</body></html>
"""


@pytest.fixture
def tool():
    return WebSearchTool()


@pytest.fixture
def mock_transport():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_DDG_HTML)

    return httpx.MockTransport(handler=handler)


def test_tool_metadata(tool):
    assert tool.name == "web_search"
    assert tool.risk_level == RiskLevel.LOW
    assert tool.description


@pytest.mark.asyncio
async def test_search_returns_results(tool, mock_transport, tmp_path):
    """Search returns title, url, and snippet for results."""
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebSearchInput(query="python asyncio"), ctx, _transport=mock_transport
    )
    assert result.is_error is False
    assert "python.org" in result.output.lower()
    assert "asyncio" in result.output.lower()


@pytest.mark.asyncio
async def test_search_respects_max_results(tool, tmp_path):
    """max_results limits the number of results returned."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=_DDG_HTML)

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebSearchInput(query="test", max_results=1), ctx, _transport=transport
    )
    assert result.is_error is False
    # Only 1 result: second URL should not be present
    assert "realpython.com" not in result.output


@pytest.mark.asyncio
async def test_search_no_results(tool, tmp_path):
    """Empty HTML returns a graceful 'no results' message."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html="<html><body><p>No results.</p></body></html>")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(WebSearchInput(query="xyzzy"), ctx, _transport=transport)
    assert result.is_error is False
    assert "no results" in result.output.lower()


@pytest.mark.asyncio
async def test_search_http_error(tool, tmp_path):
    """HTTP errors are returned as ToolResult(error=...)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(WebSearchInput(query="something"), ctx, _transport=transport)
    assert result.is_error is True
    assert "429" in result.error


@pytest.mark.asyncio
async def test_search_connection_error(tool, tmp_path):
    """Connection errors are returned gracefully."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(WebSearchInput(query="oops"), ctx, _transport=transport)
    assert result.is_error is True
