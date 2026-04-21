# Norn Phase 6: Web Search + MCP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add web browsing capabilities (fetch + search) and Model Context Protocol (MCP) client integration, expanding Norn's reach beyond the local filesystem.

**Architecture:** Two feature areas, each gated by its own flag. Web tools (`web_search` flag): `WebFetchTool` (httpx async HTTP + stdlib HTML stripper) and `WebSearchTool` (DuckDuckGo lite HTML scraping, no API key). MCP integration (`mcp` flag): `MCPToolAdapter` wraps each MCP server tool as a Norn `Tool` using a dynamic Pydantic model that reports the server's inputSchema; `load_mcp_tools()` connects to configured servers at startup and injects adapters into the registry. Both are registered in `_build_registry()` with feature-flag gating. `httpx` is already a transitive dependency via litellm (added explicitly to `[web]` group); `mcp>=1.0` goes in `[mcp]` group.

**Tech Stack:** Python 3.11+, asyncio, Pydantic v2, httpx, mcp 1.26+, pytest, ruff. New optional deps: `httpx>=0.27` (web), `mcp>=1.0` (mcp).

**Tools:**

| # | Tool / Component | Risk | Purpose |
|---|---------|------|---------|
| 1 | `web_fetch` | LOW | Fetch a URL, strip HTML, return text (up to max_chars) |
| 2 | `web_search` | LOW | DuckDuckGo search, return title/url/snippet for top N results |
| 3 | `MCPToolAdapter` | MEDIUM | Wraps an MCP server tool as a Norn Tool with dynamic schema |
| 4 | `load_mcp_tools` | — | Async helper: connects to all MCP servers, returns adapters |

---

## Task 0: Optional Dependencies and Directory Setup

**Files:**
- Modify: `pyproject.toml`
- Create: `src/norn/tools/web/__init__.py`
- Create: `tests/test_tools/test_web/__init__.py`

**Step 1: Create directories**

```bash
mkdir -p src/norn/tools/web tests/test_tools/test_web
touch src/norn/tools/web/__init__.py tests/test_tools/test_web/__init__.py
```

**Step 2: Add optional dependency groups to `pyproject.toml`**

After the existing `ml` optional-dependencies block, add:

```toml
web = [
    "httpx>=0.27",
]
mcp_tools = [
    "mcp>=1.0",
]
```

Note: name is `mcp_tools` (not `mcp`) to avoid conflict with the `mcp` package name.

**Step 3: Verify httpx is already available (transitive dep via litellm)**

Run: `uv run python -c "import httpx; print('httpx', httpx.__version__)"`
Expected: prints httpx version (e.g. `httpx 0.28.1`)

**Step 4: Verify mcp is already installed**

Run: `uv run python -c "from mcp import ClientSession; from mcp.client.stdio import stdio_client; print('mcp OK')"`
Expected: `mcp OK`

**Step 5: Run existing tests (no regression)**

Run: `uv run pytest tests/ -q --tb=short`
Expected: 311 passed, 0 failed

**Step 6: Commit**

```bash
git add pyproject.toml src/norn/tools/web/__init__.py tests/test_tools/test_web/__init__.py
git commit -m "chore: add web and mcp optional dependency groups and directory structure"
```

---

## Task 1: WebFetchTool

Fetches a URL with httpx (async) and returns the text content. If the response is HTML, strips tags using Python's built-in `html.parser`. Truncates at `max_chars` to avoid flooding the LLM context.

**Files:**
- Create: `src/norn/tools/web/web_fetch.py`
- Create: `tests/test_tools/test_web/test_web_fetch.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_web/test_web_fetch.py
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
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_web/test_web_fetch.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.tools.web.web_fetch'`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/web/web_fetch.py
"""Web fetch tool — fetches a URL and returns its text content."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult

if TYPE_CHECKING:
    import httpx


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
                error="httpx is not installed. Install with: uv pip install 'norn[web]'"
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
                    error=f"HTTP {response.status_code}: {response.reason_phrase}"
                )

            content_type = response.headers.get("content-type", "")
            text = response.text

            if "html" in content_type or text.lstrip().startswith("<"):
                text = _strip_html(text)

            if len(text) > input.max_chars:
                text = text[: input.max_chars] + f"\n\n[Truncated at {input.max_chars} chars]"

            return ToolResult(output=text or "(empty response)")

        except Exception as e:
            return ToolResult(error=f"Fetch failed: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_web/test_web_fetch.py -v`
Expected: 6 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/web/web_fetch.py`

**Step 6: Commit**

```bash
git add src/norn/tools/web/web_fetch.py tests/test_tools/test_web/test_web_fetch.py
git commit -m "feat(web): add web_fetch tool for URL fetching with HTML stripping"
```

---

## Task 2: WebSearchTool

Searches DuckDuckGo Lite (POST to `https://lite.duckduckgo.com/lite/`) and returns top N results with title, URL, and snippet. Uses stdlib HTML parsing — no extra dependencies.

**Files:**
- Create: `src/norn/tools/web/web_search.py`
- Create: `tests/test_tools/test_web/test_web_search.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_web/test_web_search.py
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
    result = await tool.execute(
        WebSearchInput(query="xyzzy"), ctx, _transport=transport
    )
    assert result.is_error is False
    assert "no results" in result.output.lower()


@pytest.mark.asyncio
async def test_search_http_error(tool, tmp_path):
    """HTTP errors are returned as ToolResult(error=...)."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, text="Too Many Requests")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebSearchInput(query="something"), ctx, _transport=transport
    )
    assert result.is_error is True
    assert "429" in result.error


@pytest.mark.asyncio
async def test_search_connection_error(tool, tmp_path):
    """Connection errors are returned gracefully."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler=handler)
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        WebSearchInput(query="oops"), ctx, _transport=transport
    )
    assert result.is_error is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_tools/test_web/test_web_search.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Write minimal implementation**

```python
# src/norn/tools/web/web_search.py
"""Web search tool — searches DuckDuckGo Lite, no API key required."""

from __future__ import annotations

from html.parser import HTMLParser
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult

if TYPE_CHECKING:
    import httpx

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
    try:
        parser.feed(html)
    except Exception:
        pass
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
                error="httpx is not installed. Install with: uv pip install 'norn[web]'"
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
                    error=f"HTTP {response.status_code}: {response.reason_phrase}"
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
            return ToolResult(error=f"Search failed: {e}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_tools/test_web/test_web_search.py -v`
Expected: 5 PASSED

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/tools/web/web_search.py`

**Step 6: Commit**

```bash
git add src/norn/tools/web/web_search.py tests/test_tools/test_web/test_web_search.py
git commit -m "feat(web): add web_search tool using DuckDuckGo Lite"
```

---

## Task 3: `web_search` Feature Flag + CLI Wiring

Wire web tools behind the `web_search` feature flag. Update `FlagsConfig`, `_build_flag_registry`, `_build_registry`, and the `config` CLI command.

**Files:**
- Modify: `src/norn/core/config.py`
- Modify: `src/norn/cli/main.py`
- Modify: `configs/default.yaml`
- Create: `tests/test_tools/test_web/test_web_registration.py`

**Step 1: Write the failing tests**

```python
# tests/test_tools/test_web/test_web_registration.py
"""Tests for web tool registration."""

from norn.flags.registry import FeatureFlag, FeatureFlagRegistry
from norn.tools.registry import ToolRegistry
from norn.tools.web.web_fetch import WebFetchTool
from norn.tools.web.web_search import WebSearchTool


def _build_web_registry(enabled: bool) -> ToolRegistry:
    flag_registry = FeatureFlagRegistry(
        flags={"web_search": FeatureFlag("web_search", enabled, "Web tools")}
    )
    flag_registry.apply_config({"web_search": enabled})
    registry = ToolRegistry(flag_registry=flag_registry)
    registry.register(WebFetchTool(), feature_flag="web_search")
    registry.register(WebSearchTool(), feature_flag="web_search")
    return registry


def test_web_tools_enabled():
    registry = _build_web_registry(enabled=True)
    names = {t.name for t in registry.list_tools()}
    assert "web_fetch" in names
    assert "web_search" in names


def test_web_tools_disabled():
    registry = _build_web_registry(enabled=False)
    names = {t.name for t in registry.list_tools()}
    assert "web_fetch" not in names
    assert "web_search" not in names


def test_web_tools_schemas_when_enabled():
    registry = _build_web_registry(enabled=True)
    schema_names = {s["name"] for s in registry.get_schemas()}
    assert "web_fetch" in schema_names
    assert "web_search" in schema_names
```

**Step 2: Run tests to verify they pass (registration is pure config, tools already exist)**

Run: `uv run pytest tests/test_tools/test_web/test_web_registration.py -v`
Expected: 3 PASSED (tools exist, just testing flag gating)

**Step 3: Update `FlagsConfig` in `src/norn/core/config.py`**

Add `web_search: bool = False` to `FlagsConfig`:

```python
class FlagsConfig(BaseModel):
    dream_system: bool = False
    coordinator: bool = False
    ml_tools: bool = False
    web_search: bool = False
```

**Step 4: Update `configs/default.yaml`**

Add under `flags:`:
```yaml
flags:
  dream_system: false   # Phase 3
  coordinator: false    # Phase 4
  ml_tools: true        # Phase 5: PyTorch/Polars/sklearn tools
  web_search: false     # Phase 6: DuckDuckGo search + web fetch
```

**Step 5: Update `src/norn/cli/main.py`**

a) Add web tool imports after the ML tool imports:

```python
from norn.tools.web.web_fetch import WebFetchTool
from norn.tools.web.web_search import WebSearchTool
```

b) In `_build_registry()`, after the ML tools block, add:

```python
    # Web tools (gated behind web_search feature flag)
    registry.register(WebFetchTool(), feature_flag="web_search")
    registry.register(WebSearchTool(), feature_flag="web_search")
```

c) In `_build_flag_registry()`, add `web_search` to the flags dict:

```python
registry = FeatureFlagRegistry(
    flags={
        "dream_system": FeatureFlag("dream_system", False, "Memory consolidation"),
        "coordinator": FeatureFlag("coordinator", False, "Multi-agent mode"),
        "ml_tools": FeatureFlag("ml_tools", True, "MLOps-specific tools"),
        "web_search": FeatureFlag("web_search", False, "Web search and fetch"),
    }
)
registry.apply_config(
    {
        "dream_system": config.flags.dream_system,
        "coordinator": config.flags.coordinator,
        "ml_tools": config.flags.ml_tools,
        "web_search": config.flags.web_search,
    }
)
```

d) In the `config()` CLI command, add after the ML tools line:

```python
    console.print(f"  Web search:   {'enabled' if cfg.flags.web_search else 'disabled'}")
```

**Step 6: Run all tests**

Run: `uv run pytest tests/ -q --tb=short`
Expected: 314 passed (311 + 3 new)

**Step 7: Commit**

```bash
git add src/norn/core/config.py src/norn/cli/main.py configs/default.yaml tests/test_tools/test_web/test_web_registration.py
git commit -m "feat(web): add web_search flag, wire web tools into CLI registry"
```

---

## Task 4: MCPServerConfig and MCPConfig Models

Define the Pydantic models for MCP server configuration. Add `MCPConfig` to `NornConfig` and update `default.yaml`.

**Files:**
- Create: `src/norn/mcp/__init__.py`
- Create: `src/norn/mcp/models.py`
- Create: `tests/test_mcp/__init__.py`
- Create: `tests/test_mcp/test_models.py`
- Modify: `src/norn/core/config.py`
- Modify: `configs/default.yaml`

**Step 1: Create directories**

```bash
mkdir -p src/norn/mcp tests/test_mcp
touch src/norn/mcp/__init__.py tests/test_mcp/__init__.py
```

**Step 2: Write the failing tests**

```python
# tests/test_mcp/test_models.py
"""Tests for MCP config models."""

import pytest

from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport


def test_mcp_transport_values():
    assert MCPTransport.STDIO == "stdio"
    assert MCPTransport.SSE == "sse"


def test_server_config_stdio():
    cfg = MCPServerConfig(
        name="filesystem",
        transport=MCPTransport.STDIO,
        command=["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
    )
    assert cfg.name == "filesystem"
    assert cfg.transport == MCPTransport.STDIO
    assert cfg.command == ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
    assert cfg.url is None
    assert cfg.env == {}


def test_server_config_sse():
    cfg = MCPServerConfig(
        name="remote",
        transport=MCPTransport.SSE,
        url="http://localhost:8000/sse",
    )
    assert cfg.url == "http://localhost:8000/sse"
    assert cfg.command is None


def test_server_config_validation_stdio_needs_command():
    """STDIO transport without command should fail validation."""
    with pytest.raises(ValueError, match="command"):
        MCPServerConfig(name="bad", transport=MCPTransport.STDIO)


def test_server_config_validation_sse_needs_url():
    """SSE transport without url should fail validation."""
    with pytest.raises(ValueError, match="url"):
        MCPServerConfig(name="bad", transport=MCPTransport.SSE)


def test_mcp_config_defaults():
    cfg = MCPConfig()
    assert cfg.servers == []
    assert cfg.enabled is False
    assert cfg.timeout > 0


def test_mcp_config_with_servers():
    cfg = MCPConfig(
        enabled=True,
        servers=[
            MCPServerConfig(
                name="fs",
                transport=MCPTransport.STDIO,
                command=["python", "-m", "myserver"],
            )
        ],
    )
    assert cfg.enabled is True
    assert len(cfg.servers) == 1
    assert cfg.servers[0].name == "fs"


def test_norn_config_has_mcp():
    """NornConfig should include mcp sub-config."""
    from norn.core.config import NornConfig

    cfg = NornConfig()
    assert hasattr(cfg, "mcp")
    assert isinstance(cfg.mcp, MCPConfig)
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.mcp'`

**Step 4: Write `src/norn/mcp/models.py`**

```python
# src/norn/mcp/models.py
"""MCP configuration models."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, model_validator


class MCPTransport(StrEnum):
    """MCP server transport protocol."""

    STDIO = "stdio"
    SSE = "sse"


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    name: str
    transport: MCPTransport
    command: list[str] | None = None  # Required for STDIO transport
    url: str | None = None            # Required for SSE transport
    env: dict[str, str] = {}

    @model_validator(mode="after")
    def _validate_transport_fields(self) -> MCPServerConfig:
        if self.transport == MCPTransport.STDIO and not self.command:
            raise ValueError("STDIO transport requires 'command' to be set")
        if self.transport == MCPTransport.SSE and not self.url:
            raise ValueError("SSE transport requires 'url' to be set")
        return self


class MCPConfig(BaseModel):
    """Configuration for the MCP client pool."""

    enabled: bool = False
    servers: list[MCPServerConfig] = []
    timeout: float = 30.0
```

**Step 5: Update `src/norn/core/config.py`**

Add the `MCPConfig` import and field to `NornConfig`. At the top of the file, in the `from __future__` section add:

After the existing `CoordinatorConfig` class and before `NornConfig`, insert nothing — just add to `NornConfig`:

```python
# Add import at top of config.py
from norn.mcp.models import MCPConfig

# Update NornConfig class:
class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()
    memory: MemorySystemConfig = MemorySystemConfig()
    coordinator: CoordinatorConfig = CoordinatorConfig()
    mcp: MCPConfig = MCPConfig()
```

Note: The import of `MCPConfig` from `norn.mcp.models` must be added after `from __future__ import annotations` and other stdlib imports. Since `config.py` uses `from __future__ import annotations`, add it inside a `TYPE_CHECKING` block OR as a direct import. Use a direct import (not TYPE_CHECKING) since it's used at runtime as a field default.

**Step 6: Update `configs/default.yaml`**

Add at the end of the file:

```yaml
mcp:
  enabled: false        # Phase 6: MCP server integration
  servers: []
  timeout: 30.0
```

**Step 7: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp/test_models.py -v`
Expected: 9 PASSED

**Step 8: Run ruff**

Run: `uv run ruff check src/norn/mcp/models.py src/norn/core/config.py`

**Step 9: Run full test suite (no regression)**

Run: `uv run pytest tests/ -q --tb=short`
Expected: All passing

**Step 10: Commit**

```bash
git add src/norn/mcp/__init__.py src/norn/mcp/models.py tests/test_mcp/__init__.py tests/test_mcp/test_models.py src/norn/core/config.py configs/default.yaml
git commit -m "feat(mcp): add MCPServerConfig and MCPConfig models, wire into NornConfig"
```

---

## Task 5: MCPToolAdapter

Wraps each MCP server tool as a Norn `Tool`. Uses a dynamic Pydantic model (class created per tool) that overrides `model_json_schema()` to return the MCP tool's `inputSchema` directly. Connects to the server, calls the tool, and disconnects — one connection per call (simple, reliable, optimizable later).

**Files:**
- Create: `src/norn/mcp/adapter.py`
- Create: `tests/test_mcp/test_adapter.py`

**Step 1: Write the failing tests**

```python
# tests/test_mcp/test_adapter.py
"""Tests for MCPToolAdapter."""

import sys
import pytest

from norn.mcp.adapter import MCPToolAdapter, _make_input_model
from norn.mcp.models import MCPServerConfig, MCPTransport
from norn.tools.base import RiskLevel, Tool, ToolContext


@pytest.fixture
def echo_server_config() -> MCPServerConfig:
    """Config for a simple echo MCP server."""
    return MCPServerConfig(
        name="echo-server",
        transport=MCPTransport.STDIO,
        command=[
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "app = FastMCP('echo')\n"
                "@app.tool()\n"
                "def echo(text: str) -> str:\n"
                "    return text\n"
                "asyncio.run(app.run_stdio_async())\n"
            ),
        ],
    )


def test_make_input_model_has_correct_schema():
    """_make_input_model returns a class whose model_json_schema matches the given schema."""
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo"}},
        "required": ["text"],
    }
    Model = _make_input_model("echo", schema)
    assert Model.model_json_schema() == schema


def test_make_input_model_accepts_extra_fields():
    """Dynamic model should accept arbitrary fields (for unknown MCP tool params)."""
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    Model = _make_input_model("tool", schema)
    instance = Model(x=1, y="extra")  # 'y' is extra, should be accepted
    assert instance.model_dump()["x"] == 1


def test_adapter_implements_tool_protocol(echo_server_config):
    """MCPToolAdapter must satisfy the Tool protocol."""
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo text back",
        input_schema={"type": "object", "properties": {"text": {"type": "string"}}},
    )
    assert isinstance(adapter, Tool)
    assert adapter.name == "echo"
    assert adapter.risk_level == RiskLevel.MEDIUM
    assert adapter.description == "Echo text back"


def test_adapter_schema_matches_mcp_schema(echo_server_config):
    """The adapter's input_model.model_json_schema() returns the MCP tool's schema."""
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo",
        input_schema=schema,
    )
    assert adapter.input_model.model_json_schema() == schema


@pytest.mark.asyncio
async def test_adapter_execute_stdio(echo_server_config, tmp_path):
    """MCPToolAdapter.execute() calls the MCP tool via stdio and returns its output."""
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}
    adapter = MCPToolAdapter(
        server_config=echo_server_config,
        tool_name="echo",
        tool_description="Echo text",
        input_schema=schema,
    )
    ctx = ToolContext(cwd=str(tmp_path))
    InputModel = adapter.input_model
    result = await adapter.execute(InputModel(text="hello mcp"), ctx)
    assert result.is_error is False
    assert "hello mcp" in result.output


@pytest.mark.asyncio
async def test_adapter_execute_bad_server(tmp_path):
    """Adapter returns ToolResult(error=...) if server fails to start."""
    bad_config = MCPServerConfig(
        name="bad",
        transport=MCPTransport.STDIO,
        command=["python", "-c", "raise SystemExit(1)"],
    )
    schema = {"type": "object", "properties": {}}
    adapter = MCPToolAdapter(
        server_config=bad_config,
        tool_name="any_tool",
        tool_description="Should fail",
        input_schema=schema,
    )
    ctx = ToolContext(cwd=str(tmp_path))
    InputModel = adapter.input_model
    result = await adapter.execute(InputModel(), ctx)
    assert result.is_error is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp/test_adapter.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.mcp.adapter'`

**Step 3: Write minimal implementation**

```python
# src/norn/mcp/adapter.py
"""MCPToolAdapter — wraps a single MCP tool as a Norn Tool."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from norn.mcp.models import MCPServerConfig, MCPTransport
from norn.tools.base import RiskLevel, ToolContext, ToolResult

if TYPE_CHECKING:
    pass


def _make_input_model(tool_name: str, schema: dict[str, Any]) -> type[BaseModel]:
    """Create a Pydantic model that reports the given JSON Schema via model_json_schema()."""
    _schema = schema  # capture in closure

    class _MCPInput(BaseModel):
        model_config = ConfigDict(extra="allow")

        @classmethod
        def model_json_schema(cls, **kwargs: Any) -> dict[str, Any]:
            return _schema

    _MCPInput.__name__ = f"MCPInput_{tool_name}"
    _MCPInput.__qualname__ = f"MCPInput_{tool_name}"
    return _MCPInput


class MCPToolAdapter:
    """Wraps an MCP server tool as a Norn Tool."""

    risk_level = RiskLevel.MEDIUM

    def __init__(
        self,
        server_config: MCPServerConfig,
        tool_name: str,
        tool_description: str,
        input_schema: dict[str, Any],
    ) -> None:
        self.name = tool_name
        self.description = tool_description
        self._server_config = server_config
        self.input_model = _make_input_model(tool_name, input_schema)

    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult:
        try:
            from mcp import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client
            from mcp.client.sse import sse_client
        except ImportError:
            return ToolResult(
                error="mcp is not installed. Install with: uv pip install 'norn[mcp_tools]'"
            )

        # Extract arguments from input model (includes extra fields)
        arguments = {k: v for k, v in input.model_dump().items() if v is not None}

        try:
            if self._server_config.transport == MCPTransport.STDIO:
                cmd = self._server_config.command or []
                params = StdioServerParameters(
                    command=cmd[0],
                    args=cmd[1:],
                    env=self._server_config.env or None,
                )
                async with stdio_client(params) as (r, w):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(self.name, arguments)
            else:
                url = self._server_config.url or ""
                async with sse_client(url) as (r, w):
                    async with ClientSession(r, w) as session:
                        await session.initialize()
                        result = await session.call_tool(self.name, arguments)

            # Extract text content from MCP result
            texts = []
            for content_item in result.content:
                if hasattr(content_item, "text"):
                    texts.append(content_item.text)
                elif hasattr(content_item, "data"):
                    texts.append(f"[binary data: {len(content_item.data)} bytes]")

            return ToolResult(output="\n".join(texts) if texts else "(empty result)")

        except Exception as e:
            return ToolResult(error=f"MCP call failed ({self._server_config.name}/{self.name}): {e}")
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp/test_adapter.py -v`
Expected: 6 PASSED

Note: `test_adapter_execute_stdio` launches a real subprocess — it may take 1-3 seconds.

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/mcp/adapter.py`

**Step 6: Commit**

```bash
git add src/norn/mcp/adapter.py tests/test_mcp/test_adapter.py
git commit -m "feat(mcp): add MCPToolAdapter wrapping MCP tools as Norn Tools"
```

---

## Task 6: `load_mcp_tools` Pool Function

Async helper that connects to all configured MCP servers, lists their tools, and returns a list of `MCPToolAdapter` instances ready for registration. Errors on individual servers are logged but don't crash the startup.

**Files:**
- Create: `src/norn/mcp/pool.py`
- Create: `tests/test_mcp/test_pool.py`

**Step 1: Write the failing tests**

```python
# tests/test_mcp/test_pool.py
"""Tests for MCP pool / load_mcp_tools."""

import sys
import pytest

from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport
from norn.mcp.pool import load_mcp_tools


def _echo_server_config(name: str = "echo") -> MCPServerConfig:
    return MCPServerConfig(
        name=name,
        transport=MCPTransport.STDIO,
        command=[
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "app = FastMCP('echo')\n"
                "@app.tool()\n"
                "def echo(text: str) -> str:\n"
                "    return text\n"
                "@app.tool()\n"
                "def add(a: int, b: int) -> int:\n"
                "    return a + b\n"
                "asyncio.run(app.run_stdio_async())\n"
            ),
        ],
    )


@pytest.mark.asyncio
async def test_load_tools_from_one_server():
    """Returns adapters for all tools in a server."""
    config = MCPConfig(enabled=True, servers=[_echo_server_config()])
    adapters = await load_mcp_tools(config)
    names = {a.name for a in adapters}
    assert "echo" in names
    assert "add" in names


@pytest.mark.asyncio
async def test_load_tools_from_multiple_servers():
    """Returns tools from all servers combined."""
    config = MCPConfig(
        enabled=True,
        servers=[_echo_server_config("s1"), _echo_server_config("s2")],
    )
    adapters = await load_mcp_tools(config)
    # Both servers expose echo + add = 4 adapters total
    assert len(adapters) == 4


@pytest.mark.asyncio
async def test_load_tools_bad_server_doesnt_crash():
    """A failing server is skipped; other servers still load."""
    bad = MCPServerConfig(
        name="bad",
        transport=MCPTransport.STDIO,
        command=["python", "-c", "raise SystemExit(1)"],
    )
    config = MCPConfig(enabled=True, servers=[bad, _echo_server_config("good")])
    adapters = await load_mcp_tools(config)
    # bad server fails silently; good server loads
    names = {a.name for a in adapters}
    assert "echo" in names


@pytest.mark.asyncio
async def test_load_tools_empty_config():
    """Empty server list returns empty adapter list."""
    config = MCPConfig(enabled=True, servers=[])
    adapters = await load_mcp_tools(config)
    assert adapters == []


@pytest.mark.asyncio
async def test_load_tools_disabled_config():
    """Disabled MCPConfig returns empty list without connecting."""
    config = MCPConfig(enabled=False, servers=[_echo_server_config()])
    adapters = await load_mcp_tools(config)
    assert adapters == []
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp/test_pool.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'norn.mcp.pool'`

**Step 3: Write minimal implementation**

```python
# src/norn/mcp/pool.py
"""MCP pool — loads tool adapters from all configured MCP servers."""

from __future__ import annotations

import logging

from norn.mcp.adapter import MCPToolAdapter
from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport

logger = logging.getLogger(__name__)


async def load_mcp_tools(config: MCPConfig) -> list[MCPToolAdapter]:
    """Connect to all MCP servers and return a flat list of MCPToolAdapter instances.

    Servers that fail to connect or list tools are skipped with a warning.
    """
    if not config.enabled:
        return []

    try:
        from mcp import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client
        from mcp.client.sse import sse_client
    except ImportError:
        logger.warning("mcp package not installed — MCP tools unavailable")
        return []

    adapters: list[MCPToolAdapter] = []

    for server in config.servers:
        try:
            tools = await _list_tools(server, config.timeout)
            for mcp_tool in tools:
                adapter = MCPToolAdapter(
                    server_config=server,
                    tool_name=mcp_tool.name,
                    tool_description=mcp_tool.description or mcp_tool.name,
                    input_schema=mcp_tool.inputSchema,
                )
                adapters.append(adapter)
        except Exception as exc:
            logger.warning("Failed to load tools from MCP server '%s': %s", server.name, exc)

    return adapters


async def _list_tools(server: MCPServerConfig, timeout: float) -> list:
    """Connect to a server, list its tools, then disconnect."""
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client
    from mcp.client.sse import sse_client

    if server.transport == MCPTransport.STDIO:
        cmd = server.command or []
        params = StdioServerParameters(
            command=cmd[0],
            args=cmd[1:],
            env=server.env or None,
        )
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
    else:
        url = server.url or ""
        async with sse_client(url) as (r, w):
            async with ClientSession(r, w) as session:
                await session.initialize()
                result = await session.list_tools()
                return result.tools
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp/test_pool.py -v`
Expected: 5 PASSED (some tests launch subprocesses — may take 5-10s)

**Step 5: Run ruff**

Run: `uv run ruff check src/norn/mcp/pool.py`

**Step 6: Commit**

```bash
git add src/norn/mcp/pool.py tests/test_mcp/test_pool.py
git commit -m "feat(mcp): add load_mcp_tools pool function for dynamic tool loading"
```

---

## Task 7: `mcp` Feature Flag + CLI Integration

Wire MCP into the CLI: add `mcp` flag to `FlagsConfig`, register it in `_build_flag_registry`, load MCP tools at startup in each command that builds a registry, and display MCP status in `norn config`.

**Files:**
- Modify: `src/norn/core/config.py`
- Modify: `src/norn/cli/main.py`
- Modify: `configs/default.yaml`
- Create: `tests/test_mcp/test_cli_integration.py`

**Step 1: Write the tests**

```python
# tests/test_mcp/test_cli_integration.py
"""Tests for MCP CLI integration."""

from norn.core.config import FlagsConfig, NornConfig
from norn.mcp.models import MCPConfig


def test_flags_config_has_mcp_field():
    """FlagsConfig must include the mcp flag."""
    flags = FlagsConfig()
    assert hasattr(flags, "mcp")
    assert flags.mcp is False  # default disabled


def test_norn_config_mcp_defaults():
    """NornConfig.mcp has sensible defaults."""
    cfg = NornConfig()
    assert cfg.mcp.enabled is False
    assert cfg.mcp.servers == []


def test_flags_config_mcp_can_be_enabled():
    cfg = NornConfig.model_validate({"flags": {"mcp": True}})
    assert cfg.flags.mcp is True
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp/test_cli_integration.py -v`
Expected: FAIL — `FlagsConfig has no attribute 'mcp'`

**Step 3: Update `FlagsConfig` in `src/norn/core/config.py`**

```python
class FlagsConfig(BaseModel):
    dream_system: bool = False
    coordinator: bool = False
    ml_tools: bool = False
    web_search: bool = False
    mcp: bool = False
```

**Step 4: Update `configs/default.yaml`**

Under `flags:`, add:
```yaml
  mcp: false            # Phase 6: MCP server tools
```

**Step 5: Update `src/norn/cli/main.py`**

a) Add `mcp` to `_build_flag_registry()`:

```python
registry = FeatureFlagRegistry(
    flags={
        "dream_system": FeatureFlag("dream_system", False, "Memory consolidation"),
        "coordinator": FeatureFlag("coordinator", False, "Multi-agent mode"),
        "ml_tools": FeatureFlag("ml_tools", True, "MLOps-specific tools"),
        "web_search": FeatureFlag("web_search", False, "Web search and fetch"),
        "mcp": FeatureFlag("mcp", False, "MCP server tools"),
    }
)
registry.apply_config(
    {
        "dream_system": config.flags.dream_system,
        "coordinator": config.flags.coordinator,
        "ml_tools": config.flags.ml_tools,
        "web_search": config.flags.web_search,
        "mcp": config.flags.mcp,
    }
)
```

b) Add a new helper function `_load_mcp_adapters` after `_build_registry`:

```python
def _load_mcp_adapters(config: NornConfig, flag_registry: FeatureFlagRegistry) -> list:
    """Synchronously load MCP tool adapters if the mcp flag is enabled."""
    if not flag_registry.is_enabled("mcp") or not config.mcp.servers:
        return []
    try:
        import asyncio

        from norn.mcp.pool import load_mcp_tools

        return asyncio.run(load_mcp_tools(config.mcp))
    except Exception as exc:
        console.print(f"[yellow]MCP load warning: {exc}[/yellow]")
        return []
```

c) In each command (`chat`, `run`, `coordinate`, `tools`) that calls `_build_registry`, add after it:

```python
registry = _build_registry(flag_registry)
# Load MCP adapters and inject into registry
for adapter in _load_mcp_adapters(config, flag_registry):
    try:
        registry.register(adapter)
    except ValueError:
        pass  # Ignore duplicate tool names across servers
```

d) In the `config()` command, add:

```python
    console.print(f"  MCP:          {'enabled' if cfg.flags.mcp else 'disabled'}")
    if cfg.mcp.enabled and cfg.mcp.servers:
        console.print(f"  MCP servers:  {len(cfg.mcp.servers)}")
```

**Step 6: Run tests**

Run: `uv run pytest tests/test_mcp/test_cli_integration.py -v`
Expected: 3 PASSED

**Step 7: Run full test suite**

Run: `uv run pytest tests/ -q --tb=short`
Expected: All passing

**Step 8: Commit**

```bash
git add src/norn/core/config.py src/norn/cli/main.py configs/default.yaml tests/test_mcp/test_cli_integration.py
git commit -m "feat(mcp): add mcp flag and wire MCP tool loading into CLI commands"
```

---

## Task 8: Integration Tests

End-to-end tests verifying web tools and MCP tools work together through the agent infrastructure.

**Files:**
- Create: `tests/test_tools/test_web/test_web_integration.py`
- Create: `tests/test_mcp/test_mcp_integration.py`

**Step 1: Write web integration tests**

```python
# tests/test_tools/test_web/test_web_integration.py
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
    flags = FeatureFlagRegistry(
        flags={"web_search": FeatureFlag("web_search", True, "Web tools")}
    )
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
```

**Step 2: Write MCP integration tests**

```python
# tests/test_mcp/test_mcp_integration.py
"""Integration tests for MCP tools."""

import sys
import pytest

from norn.mcp.adapter import MCPToolAdapter
from norn.mcp.models import MCPConfig, MCPServerConfig, MCPTransport
from norn.mcp.pool import load_mcp_tools
from norn.tools.base import Tool, ToolContext
from norn.tools.registry import ToolRegistry


def _echo_config() -> MCPServerConfig:
    return MCPServerConfig(
        name="echo",
        transport=MCPTransport.STDIO,
        command=[
            sys.executable,
            "-c",
            (
                "import asyncio\n"
                "from mcp.server.fastmcp import FastMCP\n"
                "app = FastMCP('echo')\n"
                "@app.tool()\n"
                "def echo(text: str) -> str:\n"
                "    return text\n"
                "asyncio.run(app.run_stdio_async())\n"
            ),
        ],
    )


@pytest.mark.asyncio
async def test_mcp_tools_injected_into_registry():
    """Loaded MCP adapters can be registered and queried."""
    config = MCPConfig(enabled=True, servers=[_echo_config()])
    adapters = await load_mcp_tools(config)

    registry = ToolRegistry()
    for adapter in adapters:
        registry.register(adapter)

    names = {t.name for t in registry.list_tools()}
    assert "echo" in names

    schemas = registry.get_schemas()
    schema_names = {s["name"] for s in schemas}
    assert "echo" in schema_names


@pytest.mark.asyncio
async def test_mcp_adapter_full_roundtrip(tmp_path):
    """Complete: load adapter → call tool → verify output."""
    config = MCPConfig(enabled=True, servers=[_echo_config()])
    adapters = await load_mcp_tools(config)
    echo_adapter = next((a for a in adapters if a.name == "echo"), None)
    assert echo_adapter is not None

    ctx = ToolContext(cwd=str(tmp_path))
    InputModel = echo_adapter.input_model
    result = await echo_adapter.execute(InputModel(text="integration test"), ctx)
    assert result.is_error is False
    assert "integration test" in result.output


@pytest.mark.asyncio
async def test_mcp_tool_satisfies_protocol():
    """MCPToolAdapter satisfies the Tool protocol."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}}
    adapter = MCPToolAdapter(
        server_config=_echo_config(),
        tool_name="echo",
        tool_description="Echo",
        input_schema=schema,
    )
    assert isinstance(adapter, Tool)
```

**Step 3: Run tests**

Run: `uv run pytest tests/test_tools/test_web/test_web_integration.py tests/test_mcp/test_mcp_integration.py -v`
Expected: All PASSED (MCP tests launch subprocesses, allow 10-15s)

**Step 4: Run ruff**

Run: `uv run ruff check tests/test_tools/test_web/test_web_integration.py tests/test_mcp/test_mcp_integration.py`

**Step 5: Commit**

```bash
git add tests/test_tools/test_web/test_web_integration.py tests/test_mcp/test_mcp_integration.py
git commit -m "test(web,mcp): add integration tests for web search and MCP tools"
```

---

## Task 9: Full Regression Test and Final Verification

Run all tests, ruff, verify tool listing, verify config output.

**Files:**
- None (verification only)

**Step 1: Run full test suite**

Run: `uv run pytest tests/ -v --tb=short -q`
Expected: 311 existing + ~35 new = ~346 PASSED, 0 FAILED

**Step 2: Ruff all new files**

Run: `uv run ruff check src/norn/tools/web/ src/norn/mcp/`

**Step 3: Ruff full codebase (exclude known E402 in main.py)**

Run: `uv run ruff check src/norn/ --exclude src/norn/cli/main.py`

**Step 4: Verify tool listing**

Run: `uv run norn tools`

Expected: 13 tools shown (web_search flag is disabled by default, so web_fetch and web_search should NOT appear — they are gated behind `web_search: false`). The 11 tools from Phase 5 should all still appear.

If you want to verify web tools appear: temporarily set `NORN_FLAG_WEB_SEARCH=true` and run again:

```bash
NORN_FLAG_WEB_SEARCH=true uv run norn tools
```

Expected: 13 tools (adds web_fetch and web_search).

**Step 5: Verify config output**

Run: `uv run norn config`

Expected output includes:
```
  Web search:   disabled
  MCP:          disabled
  ML tools:     enabled
```

**Step 6: Final commit if any cleanup needed**

```bash
git add -A
git commit -m "chore: phase 6 final cleanup and verification"
```

---

## Summary

| Task | Component | Tests | Files |
|------|-----------|-------|-------|
| 0 | Optional deps + directory setup | 0 | `pyproject.toml`, dirs |
| 1 | WebFetchTool | 6 | `tools/web/web_fetch.py` |
| 2 | WebSearchTool | 5 | `tools/web/web_search.py` |
| 3 | web_search flag + CLI | 3 | `config.py`, `main.py`, `default.yaml` |
| 4 | MCPServerConfig + MCPConfig | 9 | `mcp/models.py`, `config.py` |
| 5 | MCPToolAdapter | 6 | `mcp/adapter.py` |
| 6 | load_mcp_tools pool | 5 | `mcp/pool.py` |
| 7 | mcp flag + CLI | 3 | `config.py`, `main.py` |
| 8 | Integration tests | ~7 | `test_web_integration.py`, `test_mcp_integration.py` |
| 9 | Final verification | 0 | (none) |

**Total: ~44 new tests, 9 commits, 14 new files, 4 modified files**
**New optional deps: `httpx>=0.27` (in `[web]`), `mcp>=1.0` (in `[mcp_tools]`)**
