"""Web search tool — searches DuckDuckGo Lite, no API key required."""

from __future__ import annotations

import contextlib
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult

_DDG_LITE_URL = "https://lite.duckduckgo.com/lite/"


class WebSearchInput(BaseModel):
    """Input for web search."""

    query: str
    max_results: int = 5
    timeout: int = 20


class _DDGLiteParser(HTMLParser):
    """Parse DuckDuckGo Lite HTML to extract search results."""

    def __init__(self) -> None:
        super().__init__()
        self._in_result_link = False
        self._in_snippet = False
        self._current_href: str | None = None
        self._current_title: str = ""
        self._current_snippet: str = ""
        self.results: list[dict[str, str]] = []
        self._pending: dict[str, str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_dict = dict(attrs)
        cls = attr_dict.get("class", "") or ""

        if tag == "td" and "result-link" in cls:
            self._in_result_link = True
            self._pending = None
        elif tag == "td" and "result-snippet" in cls:
            self._in_snippet = True
        elif tag == "a" and self._in_result_link:
            href = attr_dict.get("href", "")
            if href and href.startswith("http"):
                self._current_href = href
                self._current_title = ""

    def handle_endtag(self, tag: str) -> None:
        if tag == "td":
            if self._in_result_link and self._current_href:
                self._pending = {"url": self._current_href, "title": self._current_title.strip()}
                self._in_result_link = False
                self._current_href = None
            elif self._in_snippet:
                if self._pending and self._current_snippet.strip():
                    self._pending["snippet"] = self._current_snippet.strip()
                    self.results.append(self._pending)
                    self._pending = None
                self._in_snippet = False
                self._current_snippet = ""

    def handle_data(self, data: str) -> None:
        if self._in_result_link and self._current_href is not None:
            self._current_title += data
        elif self._in_snippet:
            self._current_snippet += data


def _parse_ddg_html(html: str) -> list[dict[str, str]]:
    """Extract results from DuckDuckGo Lite HTML."""
    parser = _DDGLiteParser()
    with contextlib.suppress(Exception):
        parser.feed(html)
    return parser.results


class WebSearchTool:
    """Search the web using DuckDuckGo Lite. Returns title, URL, and snippet for each result."""

    name = "web_search"
    description = (
        "Search the web using DuckDuckGo. "
        "Returns title, URL, and snippet for top results. No API key required."
    )
    risk_level = RiskLevel.LOW
    input_model = WebSearchInput

    async def execute(
        self,
        input: WebSearchInput,
        ctx: ToolContext,
        _transport: Any = None,
    ) -> ToolResult:
        try:
            import httpx
        except ImportError:
            return ToolResult(
                error="httpx is not installed. Install with: uv pip install 'norn[web]'",
                error_type=ToolErrorType.NOT_SUPPORTED,
            )

        try:
            async with httpx.AsyncClient(
                transport=_transport,
                follow_redirects=True,
                timeout=input.timeout,
                headers={"User-Agent": "Norn/0.1 (coding agent; +https://github.com/norn-ai/norn)"},
            ) as client:
                response = await client.post(_DDG_LITE_URL, data={"q": input.query, "kl": "us-en"})

            if response.status_code >= 400:
                return ToolResult(
                    error=f"HTTP {response.status_code}: {response.reason_phrase}",
                    error_type=ToolErrorType.HTTP_ERROR,
                )

            results = _parse_ddg_html(response.text)

            if not results:
                return ToolResult(output=f"No results found for: {input.query}")

            lines: list[str] = [f"# Search results: {input.query}\n"]
            for i, r in enumerate(results[: input.max_results], 1):
                lines.append(f"## {i}. {r.get('title', '(no title)')}")
                lines.append(f"URL: {r['url']}")
                if snippet := r.get("snippet"):
                    lines.append(snippet)
                lines.append("")

            return ToolResult(output="\n".join(lines).strip())

        except Exception as e:
            return ToolResult(
                error=f"Search failed: {e}",
                error_type=ToolErrorType.NETWORK_ERROR,
            )
