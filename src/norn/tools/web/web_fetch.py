"""Web fetch tool — fetches a URL and returns its text content."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolErrorType, ToolResult


class WebFetchInput(BaseModel):
    """Input for web fetch."""

    url: str
    timeout: int = 30
    max_chars: int = 10_000


class _HTMLTextExtractor(HTMLParser):
    """Minimal HTML → plain-text extractor using stdlib."""

    _SKIP_TAGS = {"script", "style", "head", "meta", "link", "noscript"}

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag.lower() in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            stripped = data.strip()
            if stripped:
                self._parts.append(stripped)

    @property
    def text(self) -> str:
        raw = " ".join(self._parts)
        # Collapse multiple whitespace / newlines
        return re.sub(r"\s{2,}", "\n", raw).strip()


def _strip_html(html: str) -> str:
    """Extract plain text from HTML."""
    parser = _HTMLTextExtractor()
    try:
        parser.feed(html)
        return parser.text
    except Exception:
        # Fallback: strip all tags with regex
        return re.sub(r"<[^>]+>", " ", html).strip()


class WebFetchTool:
    """Fetch a URL and return its text content (HTML tags stripped)."""

    name = "web_fetch"
    description = (
        "Fetch a URL and return its text content. "
        "HTML tags are stripped. Output is truncated at max_chars."
    )
    risk_level = RiskLevel.LOW
    input_model = WebFetchInput

    async def execute(
        self,
        input: WebFetchInput,
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
            ) as client:
                response = await client.get(input.url)

            if response.status_code >= 400:
                return ToolResult(
                    error=f"HTTP {response.status_code}: {response.reason_phrase}",
                    error_type=ToolErrorType.HTTP_ERROR,
                )

            content_type = response.headers.get("content-type", "")
            text = response.text

            if "html" in content_type or text.lstrip().startswith("<"):
                text = _strip_html(text)

            if len(text) > input.max_chars:
                text = text[: input.max_chars] + f"\n\n[Truncated at {input.max_chars} chars]"

            return ToolResult(output=text or "(empty response)")

        except Exception as e:
            return ToolResult(
                error=f"Fetch failed: {e}",
                error_type=ToolErrorType.NETWORK_ERROR,
            )
