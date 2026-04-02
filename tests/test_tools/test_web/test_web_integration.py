"""Integration tests for web tools."""

import httpx
import pytest

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.base import ToolContext
from norn.tools.registry import ToolRegistry
from norn.tools.web.web_fetch import WebFetchInput, WebFetchTool
from norn.tools.web.web_search import WebSearchInput, WebSearchTool


@pytest.fixture
def web_registry():
    flags = FeatureFlagRegistry(flags={"web_search": FeatureFlag("web_search", True, "Web tools")})
    flags.apply_config({"web_search": True})
    registry = ToolRegistry(flag_registry=flags)
    registry.register(WebFetchTool(), feature_flag="web_search")
    registry.register(WebSearchTool(), feature_flag="web_search")
    return registry


def test_web_tools_have_schemas(web_registry):
    schemas = web_registry.get_schemas()
    names = {s["name"] for s in schemas}
    assert "web_fetch" in names
    assert "web_search" in names
    for schema in schemas:
        assert "parameters" in schema
        assert "description" in schema


@pytest.mark.asyncio
async def test_fetch_then_process(tmp_path):
    """Simulate: fetch URL → work with content."""
    html = "<html><body><h1>Asyncio Docs</h1><p>asyncio is the standard library.</p></body></html>"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=html)

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    tool = WebFetchTool()
    result = await tool.execute(
        WebFetchInput(url="https://docs.python.org/asyncio"), ctx, _transport=transport
    )
    assert result.is_error is False
    assert "Asyncio Docs" in result.output
    assert "asyncio" in result.output.lower()


@pytest.mark.asyncio
async def test_search_then_fetch(tmp_path):
    """Simulate: search → pick URL → fetch."""
    search_html = """\
<html><body>
<table>
  <tr><td class="result-link"><a href="https://python.org/asyncio">Python asyncio</a></td></tr>
  <tr><td class="result-snippet">The asyncio library.</td></tr>
</table>
</body></html>
"""
    fetch_html = "<html><body><p>Asyncio is great for concurrent code.</p></body></html>"

    def search_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=search_html)

    def fetch_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, html=fetch_html)

    ctx = ToolContext(cwd=str(tmp_path))

    # Step 1: search
    search_tool = WebSearchTool()
    search_result = await search_tool.execute(
        WebSearchInput(query="python asyncio"),
        ctx,
        _transport=httpx.MockTransport(handler=search_handler),
    )
    assert search_result.is_error is False
    assert "python.org" in search_result.output

    # Step 2: fetch the first URL
    fetch_tool = WebFetchTool()
    fetch_result = await fetch_tool.execute(
        WebFetchInput(url="https://python.org/asyncio"),
        ctx,
        _transport=httpx.MockTransport(handler=fetch_handler),
    )
    assert fetch_result.is_error is False
    assert "concurrent" in fetch_result.output
