# Norn Phase 1: Core Agent Loop Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the foundational core of Norn — LLM abstraction, tool system, agent loop, and CLI — so that a user can run `norn chat` and interact with an LLM that can call tools.

**Architecture:** Protocol-based, async-first Python agent. LLM providers implement a `LLMProvider` protocol. Tools implement a `Tool` protocol and register in a `ToolRegistry`. The `AgentLoop` orchestrates the message -> LLM -> tool calls -> repeat cycle. CLI via typer + rich for streaming.

**Tech Stack:** Python 3.11+, uv, asyncio, Pydantic v2, typer, rich, litellm, pytest, ruff

---

## Task 0: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/norn/__init__.py`
- Create: `src/norn/core/__init__.py`
- Create: `src/norn/tools/__init__.py`
- Create: `src/norn/cli/__init__.py`
- Create: `src/norn/permissions/__init__.py`
- Create: `src/norn/flags/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_core/__init__.py`
- Create: `tests/test_tools/__init__.py`
- Create: `configs/default.yaml`
- Create: `.gitignore`

**Step 1: Create pyproject.toml**

```toml
[project]
name = "norn"
version = "0.1.0"
description = "Open-source, model-agnostic coding agent with persistent memory and multi-agent orchestration"
requires-python = ">=3.11"
license = { text = "MIT" }
dependencies = [
    "pydantic>=2.0",
    "typer>=0.12",
    "rich>=13.0",
    "litellm>=1.50",
    "pyyaml>=6.0",
    "aiofiles>=24.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.8",
    "mypy>=1.13",
]

[project.scripts]
norn = "norn.cli.main:app"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/norn"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "F", "I", "N", "W", "UP", "B", "SIM", "TCH"]

[tool.mypy]
python_version = "3.11"
strict = true
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
venv/
dist/
*.egg-info/
.mypy_cache/
.pytest_cache/
.ruff_cache/
.env
```

**Step 3: Create default config**

```yaml
# configs/default.yaml
llm:
  provider: "ollama"
  model: "qwen2.5-coder:14b"
  fallback_provider: "openrouter"
  fallback_model: "anthropic/claude-sonnet-4-20250514"
  temperature: 0.0
  max_tokens: 4096

permissions:
  mode: "interactive"  # interactive | auto | yolo | strict

flags:
  dream_system: false   # Phase 3
  coordinator: false    # Phase 4
  ml_tools: false       # Future
```

**Step 4: Create all __init__.py files and directory structure**

All `__init__.py` files are empty for now.

**Step 5: Set up venv and install**

Run:
```bash
cd ~/norn
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```
Expected: Successful installation

**Step 6: Commit**

```bash
git add -A
git commit -m "feat: scaffold Norn project structure with pyproject.toml"
```

---

## Task 1: Core Models

**Files:**
- Create: `src/norn/core/models.py`
- Create: `tests/test_core/test_models.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_models.py
"""Tests for core message and response models."""
import pytest
from norn.core.models import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    LLMResponse,
    StreamChunk,
)


def test_message_creation():
    msg = Message(role=Role.USER, content="hello")
    assert msg.role == Role.USER
    assert msg.content == "hello"
    assert msg.tool_call_id is None


def test_tool_call_creation():
    call = ToolCall(id="call_123", name="bash", arguments={"command": "ls"})
    assert call.id == "call_123"
    assert call.name == "bash"
    assert call.arguments == {"command": "ls"}


def test_tool_result_success():
    result = ToolResult(output="file1.py\nfile2.py")
    assert result.output == "file1.py\nfile2.py"
    assert result.error is None
    assert result.is_error is False


def test_tool_result_error():
    result = ToolResult(error="Permission denied")
    assert result.output is None
    assert result.error == "Permission denied"
    assert result.is_error is True


def test_llm_response_text_only():
    resp = LLMResponse(content="Hello!", tool_calls=[])
    assert resp.content == "Hello!"
    assert resp.tool_calls == []
    assert resp.has_tool_calls is False


def test_llm_response_with_tool_calls():
    calls = [ToolCall(id="c1", name="bash", arguments={"command": "ls"})]
    resp = LLMResponse(content=None, tool_calls=calls)
    assert resp.has_tool_calls is True


def test_stream_chunk():
    chunk = StreamChunk(content="partial", done=False)
    assert chunk.content == "partial"
    assert chunk.done is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_models.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/norn/core/models.py
"""Core message and response models for the Norn agent."""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, computed_field


class Role(str, Enum):
    """Message role in the conversation."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single message in the conversation history."""
    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """A tool invocation requested by the LLM."""
    id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result of a tool execution."""
    output: str | None = None
    error: str | None = None

    @computed_field
    @property
    def is_error(self) -> bool:
        return self.error is not None


class LLMResponse(BaseModel):
    """Response from an LLM completion call."""
    content: str | None = None
    tool_calls: list[ToolCall] = []
    usage: TokenUsage | None = None

    @computed_field
    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class StreamChunk(BaseModel):
    """A chunk from a streaming LLM response."""
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    done: bool = False
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_models.py -v`
Expected: All 7 tests PASS

**Step 5: Run ruff**

Run: `ruff check src/norn/core/models.py`
Expected: No issues

**Step 6: Commit**

```bash
git add src/norn/core/models.py tests/test_core/test_models.py
git commit -m "feat(core): add message, tool call, and response models"
```

---

## Task 2: LLM Provider Protocol + LiteLLM Implementation

**Files:**
- Create: `src/norn/core/llm.py`
- Create: `tests/test_core/test_llm.py`

**Step 1: Write the failing test**

```python
# tests/test_core/test_llm.py
"""Tests for LLM provider abstraction."""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from norn.core.llm import LiteLLMProvider, build_tool_schemas
from norn.core.models import Message, Role, LLMResponse, ToolCall


@pytest.fixture
def provider():
    return LiteLLMProvider(model="ollama/qwen2.5-coder:14b")


def test_provider_creation(provider):
    assert provider.model == "ollama/qwen2.5-coder:14b"


def test_build_tool_schemas_empty():
    schemas = build_tool_schemas([])
    assert schemas == []


def test_build_tool_schemas_from_dict():
    tool_def = {
        "name": "bash",
        "description": "Run a shell command",
        "parameters": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    }
    schemas = build_tool_schemas([tool_def])
    assert len(schemas) == 1
    assert schemas[0]["type"] == "function"
    assert schemas[0]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_complete_text_response(provider):
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Hello!"
    mock_response.choices[0].message.tool_calls = None
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.usage.total_tokens = 15

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(
            messages=[Message(role=Role.USER, content="Hi")]
        )
        assert isinstance(result, LLMResponse)
        assert result.content == "Hello!"
        assert result.has_tool_calls is False


@pytest.mark.asyncio
async def test_complete_with_tool_calls(provider):
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_abc"
    mock_tool_call.function.name = "bash"
    mock_tool_call.function.arguments = '{"command": "ls"}'

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = None
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    mock_response.usage.prompt_tokens = 20
    mock_response.usage.completion_tokens = 10
    mock_response.usage.total_tokens = 30

    with patch("litellm.acompletion", new_callable=AsyncMock, return_value=mock_response):
        result = await provider.complete(
            messages=[Message(role=Role.USER, content="list files")],
            tools=[{
                "name": "bash",
                "description": "Run shell",
                "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
            }],
        )
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_llm.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write minimal implementation**

```python
# src/norn/core/llm.py
"""LLM provider abstraction for Norn."""
from __future__ import annotations

import json
from typing import AsyncIterator, Protocol, runtime_checkable

import litellm

from norn.core.models import (
    LLMResponse,
    Message,
    Role,
    StreamChunk,
    TokenUsage,
    ToolCall,
)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol that all LLM backends must implement."""

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse: ...

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]: ...


def build_tool_schemas(tools: list[dict]) -> list[dict]:
    """Convert tool definitions to OpenAI-compatible function schemas."""
    if not tools:
        return []
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tools
    ]


def _messages_to_dicts(messages: list[Message]) -> list[dict]:
    """Convert Message objects to dicts for litellm."""
    result = []
    for msg in messages:
        d: dict = {"role": msg.role.value, "content": msg.content or ""}
        if msg.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments)},
                }
                for tc in msg.tool_calls
            ]
        if msg.tool_call_id:
            d["tool_call_id"] = msg.tool_call_id
        result.append(d)
    return result


def _parse_tool_calls(raw_tool_calls: list | None) -> list[ToolCall]:
    """Parse tool calls from litellm response."""
    if not raw_tool_calls:
        return []
    calls = []
    for tc in raw_tool_calls:
        arguments = tc.function.arguments
        if isinstance(arguments, str):
            arguments = json.loads(arguments)
        calls.append(
            ToolCall(id=tc.id, name=tc.function.name, arguments=arguments)
        )
    return calls


class LiteLLMProvider:
    """LLM provider using litellm for universal model support."""

    def __init__(self, model: str, api_base: str | None = None):
        self.model = model
        self.api_base = api_base
        # Suppress litellm logging noise
        litellm.suppress_debug_info = True

    async def complete(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        kwargs: dict = {
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        tool_schemas = build_tool_schemas(tools or [])
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await litellm.acompletion(**kwargs)
        choice = response.choices[0]

        return LLMResponse(
            content=choice.message.content,
            tool_calls=_parse_tool_calls(choice.message.tool_calls),
            usage=TokenUsage(
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
            ),
        )

    async def stream(
        self,
        messages: list[Message],
        tools: list[dict] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[StreamChunk]:
        kwargs: dict = {
            "model": self.model,
            "messages": _messages_to_dicts(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if self.api_base:
            kwargs["api_base"] = self.api_base

        tool_schemas = build_tool_schemas(tools or [])
        if tool_schemas:
            kwargs["tools"] = tool_schemas

        response = await litellm.acompletion(**kwargs)
        async for chunk in response:
            delta = chunk.choices[0].delta
            yield StreamChunk(
                content=delta.content if hasattr(delta, "content") else None,
                done=chunk.choices[0].finish_reason is not None,
            )
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_core/test_llm.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/llm.py tests/test_core/test_llm.py
git commit -m "feat(core): add LLM provider protocol and LiteLLM implementation"
```

---

## Task 3: Tool Base Protocol + Registry

**Files:**
- Create: `src/norn/tools/base.py`
- Create: `src/norn/tools/registry.py`
- Create: `tests/test_tools/test_registry.py`

**Step 1: Write the failing test**

```python
# tests/test_tools/test_registry.py
"""Tests for tool base protocol and registry."""
import pytest
from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class MockInput(BaseModel):
    query: str


class MockTool:
    name = "mock_tool"
    description = "A mock tool for testing"
    risk_level = RiskLevel.LOW
    input_model = MockInput

    async def execute(self, input: MockInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"result: {input.query}")


@pytest.fixture
def registry():
    return ToolRegistry()


@pytest.fixture
def mock_tool():
    return MockTool()


def test_register_tool(registry, mock_tool):
    registry.register(mock_tool)
    assert registry.get("mock_tool") is mock_tool


def test_get_unknown_tool(registry):
    assert registry.get("nonexistent") is None


def test_list_tools(registry, mock_tool):
    registry.register(mock_tool)
    tools = registry.list_tools()
    assert len(tools) == 1
    assert tools[0].name == "mock_tool"


def test_get_schemas(registry, mock_tool):
    registry.register(mock_tool)
    schemas = registry.get_schemas()
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["name"] == "mock_tool"
    assert schema["description"] == "A mock tool for testing"
    assert "properties" in schema["parameters"]
    assert "query" in schema["parameters"]["properties"]


def test_schema_caching(registry, mock_tool):
    registry.register(mock_tool)
    schemas1 = registry.get_schemas()
    schemas2 = registry.get_schemas()
    # Same object reference = cached
    assert schemas1 is schemas2


def test_register_with_feature_flag_disabled(registry, mock_tool):
    registry.register(mock_tool, feature_flag="disabled_feature")
    # Tool registered but we can check flag status
    assert registry.get("mock_tool") is mock_tool


def test_duplicate_registration_raises(registry, mock_tool):
    registry.register(mock_tool)
    with pytest.raises(ValueError, match="already registered"):
        registry.register(mock_tool)


@pytest.mark.asyncio
async def test_tool_execution(mock_tool):
    ctx = ToolContext(cwd="/tmp")
    result = await mock_tool.execute(MockInput(query="test"), ctx)
    assert result.output == "result: test"
    assert result.is_error is False
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_tools/test_registry.py -v`
Expected: FAIL with ModuleNotFoundError

**Step 3: Write implementation**

```python
# src/norn/tools/base.py
"""Tool base types and protocols for Norn."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, computed_field


class RiskLevel(str, Enum):
    """Risk classification for tool actions."""
    LOW = "low"       # Read-only operations
    MEDIUM = "medium"  # Writes with potential undo
    HIGH = "high"      # Destructive or irreversible


class ToolContext(BaseModel):
    """Context passed to tool execution."""
    cwd: str = "."


class ToolResult(BaseModel):
    """Result of a tool execution."""
    output: str | None = None
    error: str | None = None

    @computed_field
    @property
    def is_error(self) -> bool:
        return self.error is not None


@runtime_checkable
class Tool(Protocol):
    """Protocol that all tools must implement."""
    name: str
    description: str
    risk_level: RiskLevel
    input_model: type[BaseModel]

    async def execute(self, input: BaseModel, ctx: ToolContext) -> ToolResult: ...
```

```python
# src/norn/tools/registry.py
"""Tool registry with schema caching."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from norn.tools.base import Tool


class ToolRegistry:
    """Registry for agent tools with cached JSON schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}
        self._schema_cache: list[dict[str, Any]] | None = None
        self._feature_flags: dict[str, str] = {}  # tool_name -> flag_name

    def register(self, tool: Tool, feature_flag: str | None = None) -> None:
        """Register a tool. Raises ValueError if already registered."""
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' already registered")
        self._tools[tool.name] = tool
        if feature_flag:
            self._feature_flags[tool.name] = feature_flag
        self._schema_cache = None  # Invalidate cache

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get JSON schemas for all tools. Cached for prompt efficiency."""
        if self._schema_cache is not None:
            return self._schema_cache

        schemas = []
        for tool in self._tools.values():
            schema = tool.input_model.model_json_schema()
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": schema,
            })
        self._schema_cache = schemas
        return self._schema_cache

    def scoped(self, tool_names: list[str]) -> ToolRegistry:
        """Create a new registry with only the specified tools."""
        scoped = ToolRegistry()
        for name in tool_names:
            tool = self.get(name)
            if tool:
                scoped.register(tool)
        return scoped
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_tools/test_registry.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add src/norn/tools/base.py src/norn/tools/registry.py tests/test_tools/test_registry.py
git commit -m "feat(tools): add tool protocol, risk levels, and registry with schema caching"
```

---

## Task 4: FileReadTool + GlobTool + GrepTool

**Files:**
- Create: `src/norn/tools/file_read.py`
- Create: `src/norn/tools/glob_tool.py`
- Create: `src/norn/tools/grep_tool.py`
- Create: `tests/test_tools/test_file_read.py`
- Create: `tests/test_tools/test_glob.py`
- Create: `tests/test_tools/test_grep.py`

**Step 1: Write failing tests**

```python
# tests/test_tools/test_file_read.py
"""Tests for FileReadTool."""
import pytest
from pathlib import Path

from norn.tools.file_read import FileReadTool, FileReadInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return FileReadTool()


@pytest.fixture
def tmp_file(tmp_path):
    f = tmp_path / "test.txt"
    f.write_text("line1\nline2\nline3\n")
    return f


def test_tool_metadata(tool):
    assert tool.name == "file_read"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_read_file(tool, tmp_file):
    ctx = ToolContext(cwd=str(tmp_file.parent))
    result = await tool.execute(
        FileReadInput(path=str(tmp_file)), ctx
    )
    assert "line1" in result.output
    assert "line2" in result.output
    assert result.is_error is False


@pytest.mark.asyncio
async def test_read_nonexistent(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileReadInput(path=str(tmp_path / "nope.txt")), ctx
    )
    assert result.is_error is True
    assert "not found" in result.error.lower() or "No such file" in result.error


@pytest.mark.asyncio
async def test_read_directory(tool, tmp_path):
    (tmp_path / "a.py").touch()
    (tmp_path / "b.py").touch()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileReadInput(path=str(tmp_path)), ctx
    )
    assert "a.py" in result.output
    assert "b.py" in result.output
```

```python
# tests/test_tools/test_glob.py
"""Tests for GlobTool."""
import pytest

from norn.tools.glob_tool import GlobTool, GlobInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return GlobTool()


def test_tool_metadata(tool):
    assert tool.name == "glob"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_glob_pattern(tool, tmp_path):
    (tmp_path / "foo.py").touch()
    (tmp_path / "bar.py").touch()
    (tmp_path / "baz.txt").touch()
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        GlobInput(pattern="*.py", path=str(tmp_path)), ctx
    )
    assert "foo.py" in result.output
    assert "bar.py" in result.output
    assert "baz.txt" not in result.output
```

```python
# tests/test_tools/test_grep.py
"""Tests for GrepTool."""
import pytest

from norn.tools.grep_tool import GrepTool, GrepInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return GrepTool()


def test_tool_metadata(tool):
    assert tool.name == "grep"
    assert tool.risk_level == RiskLevel.LOW


@pytest.mark.asyncio
async def test_grep_pattern(tool, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("def hello():\n    return 'world'\n\ndef goodbye():\n    pass\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        GrepInput(pattern="def \\w+", path=str(tmp_path)), ctx
    )
    assert "hello" in result.output
    assert "goodbye" in result.output


@pytest.mark.asyncio
async def test_grep_no_matches(tool, tmp_path):
    f = tmp_path / "test.py"
    f.write_text("x = 1\n")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        GrepInput(pattern="class Foo", path=str(tmp_path)), ctx
    )
    assert result.output is not None  # Empty but no error
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_file_read.py tests/test_tools/test_glob.py tests/test_tools/test_grep.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/norn/tools/file_read.py
"""File read tool for the Norn agent."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class FileReadInput(BaseModel):
    """Input for file read operations."""
    path: str
    offset: int = 0
    limit: int = 2000


class FileReadTool:
    """Read a file or list a directory."""

    name = "file_read"
    description = "Read a file's contents or list a directory's entries."
    risk_level = RiskLevel.LOW
    input_model = FileReadInput

    async def execute(self, input: FileReadInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        if not target.exists():
            return ToolResult(error=f"Path not found: {target}")

        if target.is_dir():
            entries = sorted(target.iterdir())
            listing = "\n".join(
                f"{e.name}/" if e.is_dir() else e.name for e in entries
            )
            return ToolResult(output=listing or "(empty directory)")

        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            selected = lines[input.offset : input.offset + input.limit]
            numbered = [
                f"{i + input.offset + 1}: {line}" for i, line in enumerate(selected)
            ]
            return ToolResult(output="\n".join(numbered))
        except Exception as e:
            return ToolResult(error=str(e))
```

```python
# src/norn/tools/glob_tool.py
"""Glob tool for file pattern matching."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class GlobInput(BaseModel):
    """Input for glob pattern matching."""
    pattern: str
    path: str | None = None


class GlobTool:
    """Find files matching a glob pattern."""

    name = "glob"
    description = "Find files matching a glob pattern in a directory."
    risk_level = RiskLevel.LOW
    input_model = GlobInput

    async def execute(self, input: GlobInput, ctx: ToolContext) -> ToolResult:
        base = Path(input.path) if input.path else Path(ctx.cwd)
        if not base.is_absolute():
            base = Path(ctx.cwd) / base

        if not base.exists():
            return ToolResult(error=f"Directory not found: {base}")

        matches = sorted(base.glob(input.pattern))
        if not matches:
            return ToolResult(output="No matches found.")

        output = "\n".join(str(m) for m in matches[:500])
        return ToolResult(output=output)
```

```python
# src/norn/tools/grep_tool.py
"""Grep tool for content search."""
from __future__ import annotations

import re
from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class GrepInput(BaseModel):
    """Input for content search."""
    pattern: str
    path: str | None = None
    include: str | None = None


class GrepTool:
    """Search file contents using regex patterns."""

    name = "grep"
    description = "Search file contents using regular expressions."
    risk_level = RiskLevel.LOW
    input_model = GrepInput

    async def execute(self, input: GrepInput, ctx: ToolContext) -> ToolResult:
        base = Path(input.path) if input.path else Path(ctx.cwd)
        if not base.is_absolute():
            base = Path(ctx.cwd) / base

        if not base.exists():
            return ToolResult(error=f"Path not found: {base}")

        try:
            regex = re.compile(input.pattern)
        except re.error as e:
            return ToolResult(error=f"Invalid regex: {e}")

        results: list[str] = []
        files = base.rglob(input.include or "*") if base.is_dir() else [base]

        for filepath in files:
            if not filepath.is_file():
                continue
            try:
                text = filepath.read_text(encoding="utf-8", errors="replace")
                for i, line in enumerate(text.splitlines(), 1):
                    if regex.search(line):
                        results.append(f"{filepath}:{i}: {line.strip()}")
                        if len(results) >= 200:
                            break
            except Exception:
                continue

            if len(results) >= 200:
                break

        return ToolResult(output="\n".join(results) if results else "No matches found.")
```

**Step 4: Run tests**

Run: `pytest tests/test_tools/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/norn/tools/file_read.py src/norn/tools/glob_tool.py src/norn/tools/grep_tool.py \
        tests/test_tools/test_file_read.py tests/test_tools/test_glob.py tests/test_tools/test_grep.py
git commit -m "feat(tools): add FileRead, Glob, and Grep tools"
```

---

## Task 5: FileWriteTool + FileEditTool + BashTool

**Files:**
- Create: `src/norn/tools/file_write.py`
- Create: `src/norn/tools/file_edit.py`
- Create: `src/norn/tools/bash_tool.py`
- Create: `tests/test_tools/test_file_write.py`
- Create: `tests/test_tools/test_file_edit.py`
- Create: `tests/test_tools/test_bash.py`

**Step 1: Write failing tests**

```python
# tests/test_tools/test_file_write.py
"""Tests for FileWriteTool."""
import pytest

from norn.tools.file_write import FileWriteTool, FileWriteInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return FileWriteTool()


def test_tool_metadata(tool):
    assert tool.name == "file_write"
    assert tool.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_write_new_file(tool, tmp_path):
    target = tmp_path / "new.txt"
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileWriteInput(path=str(target), content="hello world"), ctx
    )
    assert result.is_error is False
    assert target.read_text() == "hello world"


@pytest.mark.asyncio
async def test_write_creates_parents(tool, tmp_path):
    target = tmp_path / "sub" / "dir" / "file.txt"
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        FileWriteInput(path=str(target), content="nested"), ctx
    )
    assert result.is_error is False
    assert target.read_text() == "nested"
```

```python
# tests/test_tools/test_file_edit.py
"""Tests for FileEditTool."""
import pytest

from norn.tools.file_edit import FileEditTool, FileEditInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return FileEditTool()


@pytest.fixture
def source_file(tmp_path):
    f = tmp_path / "code.py"
    f.write_text("def hello():\n    return 'world'\n")
    return f


def test_tool_metadata(tool):
    assert tool.name == "file_edit"
    assert tool.risk_level == RiskLevel.MEDIUM


@pytest.mark.asyncio
async def test_edit_replacement(tool, source_file):
    ctx = ToolContext(cwd=str(source_file.parent))
    result = await tool.execute(
        FileEditInput(
            path=str(source_file),
            old_string="return 'world'",
            new_string="return 'universe'",
        ),
        ctx,
    )
    assert result.is_error is False
    assert "universe" in source_file.read_text()
    assert "world" not in source_file.read_text()


@pytest.mark.asyncio
async def test_edit_not_found(tool, source_file):
    ctx = ToolContext(cwd=str(source_file.parent))
    result = await tool.execute(
        FileEditInput(
            path=str(source_file),
            old_string="nonexistent string",
            new_string="replacement",
        ),
        ctx,
    )
    assert result.is_error is True
    assert "not found" in result.error.lower()
```

```python
# tests/test_tools/test_bash.py
"""Tests for BashTool."""
import pytest

from norn.tools.bash_tool import BashTool, BashInput
from norn.tools.base import RiskLevel, ToolContext


@pytest.fixture
def tool():
    return BashTool()


def test_tool_metadata(tool):
    assert tool.name == "bash"
    assert tool.risk_level == RiskLevel.HIGH


@pytest.mark.asyncio
async def test_bash_echo(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="echo 'hello norn'"), ctx)
    assert result.is_error is False
    assert "hello norn" in result.output


@pytest.mark.asyncio
async def test_bash_failure(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(BashInput(command="false"), ctx)
    assert result.is_error is True


@pytest.mark.asyncio
async def test_bash_timeout(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(
        BashInput(command="sleep 10", timeout=1), ctx
    )
    assert result.is_error is True
    assert "timeout" in result.error.lower()
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_tools/test_file_write.py tests/test_tools/test_file_edit.py tests/test_tools/test_bash.py -v`
Expected: FAIL

**Step 3: Write implementations**

```python
# src/norn/tools/file_write.py
"""File write tool for the Norn agent."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class FileWriteInput(BaseModel):
    """Input for file write operations."""
    path: str
    content: str


class FileWriteTool:
    """Create or overwrite a file."""

    name = "file_write"
    description = "Create a new file or overwrite an existing one with the provided content."
    risk_level = RiskLevel.MEDIUM
    input_model = FileWriteInput

    async def execute(self, input: FileWriteInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(input.content, encoding="utf-8")
            return ToolResult(output=f"Wrote {len(input.content)} bytes to {target}")
        except Exception as e:
            return ToolResult(error=str(e))
```

```python
# src/norn/tools/file_edit.py
"""File edit tool for targeted string replacement."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class FileEditInput(BaseModel):
    """Input for file edit operations."""
    path: str
    old_string: str
    new_string: str
    replace_all: bool = False


class FileEditTool:
    """Replace a specific string in a file."""

    name = "file_edit"
    description = "Replace an exact string occurrence in a file with new content."
    risk_level = RiskLevel.MEDIUM
    input_model = FileEditInput

    async def execute(self, input: FileEditInput, ctx: ToolContext) -> ToolResult:
        target = Path(input.path)
        if not target.is_absolute():
            target = Path(ctx.cwd) / target

        if not target.exists():
            return ToolResult(error=f"File not found: {target}")

        try:
            content = target.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult(error=f"Cannot read file: {e}")

        count = content.count(input.old_string)

        if count == 0:
            return ToolResult(error=f"Old string not found in {target}")

        if count > 1 and not input.replace_all:
            return ToolResult(
                error=f"Found {count} matches. Use replace_all=true or provide more context."
            )

        if input.replace_all:
            new_content = content.replace(input.old_string, input.new_string)
        else:
            new_content = content.replace(input.old_string, input.new_string, 1)

        target.write_text(new_content, encoding="utf-8")
        replaced = count if input.replace_all else 1
        return ToolResult(output=f"Replaced {replaced} occurrence(s) in {target}")
```

```python
# src/norn/tools/bash_tool.py
"""Bash tool for shell command execution."""
from __future__ import annotations

import asyncio

from pydantic import BaseModel

from norn.tools.base import RiskLevel, ToolContext, ToolResult


class BashInput(BaseModel):
    """Input for bash command execution."""
    command: str
    timeout: int = 120


class BashTool:
    """Execute a shell command."""

    name = "bash"
    description = "Execute a bash command in the shell."
    risk_level = RiskLevel.HIGH
    input_model = BashInput

    async def execute(self, input: BashInput, ctx: ToolContext) -> ToolResult:
        try:
            process = await asyncio.create_subprocess_shell(
                input.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=ctx.cwd,
            )
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=input.timeout
            )

            output = stdout.decode("utf-8", errors="replace")
            errors = stderr.decode("utf-8", errors="replace")

            if process.returncode != 0:
                return ToolResult(
                    error=f"Exit code {process.returncode}\n{errors or output}"
                )

            combined = output
            if errors:
                combined += f"\nSTDERR:\n{errors}"
            return ToolResult(output=combined)

        except asyncio.TimeoutError:
            process.kill()
            return ToolResult(error=f"Command timed out after {input.timeout}s")
        except Exception as e:
            return ToolResult(error=str(e))
```

**Step 4: Run all tool tests**

Run: `pytest tests/test_tools/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add src/norn/tools/file_write.py src/norn/tools/file_edit.py src/norn/tools/bash_tool.py \
        tests/test_tools/test_file_write.py tests/test_tools/test_file_edit.py tests/test_tools/test_bash.py
git commit -m "feat(tools): add FileWrite, FileEdit, and Bash tools"
```

---

## Task 6: Config System

**Files:**
- Create: `src/norn/core/config.py`
- Create: `tests/test_core/test_config.py`

**Step 1: Write failing test**

```python
# tests/test_core/test_config.py
"""Tests for configuration system."""
import pytest
from pathlib import Path

from norn.core.config import NornConfig, LLMConfig, PermissionMode


def test_default_config():
    config = NornConfig()
    assert config.llm.provider == "ollama"
    assert config.permissions.mode == PermissionMode.INTERACTIVE


def test_config_from_yaml(tmp_path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "llm:\n  provider: openrouter\n  model: anthropic/claude-sonnet-4-20250514\n"
        "permissions:\n  mode: auto\n"
    )
    config = NornConfig.from_yaml(config_file)
    assert config.llm.provider == "openrouter"
    assert config.llm.model == "anthropic/claude-sonnet-4-20250514"
    assert config.permissions.mode == PermissionMode.AUTO


def test_config_env_override(monkeypatch):
    monkeypatch.setenv("NORN_LLM_MODEL", "gpt-4o")
    config = NornConfig()
    config.apply_env_overrides()
    assert config.llm.model == "gpt-4o"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_config.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/norn/core/config.py
"""Configuration system for Norn."""
from __future__ import annotations

import os
from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel


class PermissionMode(str, Enum):
    INTERACTIVE = "interactive"
    AUTO = "auto"
    YOLO = "yolo"
    STRICT = "strict"


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "qwen2.5-coder:14b"
    fallback_provider: str | None = "openrouter"
    fallback_model: str | None = None
    temperature: float = 0.0
    max_tokens: int = 4096
    api_base: str | None = None


class PermissionsConfig(BaseModel):
    mode: PermissionMode = PermissionMode.INTERACTIVE


class FlagsConfig(BaseModel):
    dream_system: bool = False
    coordinator: bool = False
    ml_tools: bool = False


class NornConfig(BaseModel):
    llm: LLMConfig = LLMConfig()
    permissions: PermissionsConfig = PermissionsConfig()
    flags: FlagsConfig = FlagsConfig()

    @classmethod
    def from_yaml(cls, path: Path) -> NornConfig:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def load(cls) -> NornConfig:
        """Load config with fallback: project -> user -> defaults."""
        candidates = [
            Path.cwd() / ".norn" / "config.yaml",
            Path.home() / ".norn" / "config.yaml",
            Path(__file__).parent.parent.parent.parent / "configs" / "default.yaml",
        ]
        for path in candidates:
            if path.exists():
                return cls.from_yaml(path)
        return cls()

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        env_map = {
            "NORN_LLM_PROVIDER": ("llm", "provider"),
            "NORN_LLM_MODEL": ("llm", "model"),
            "NORN_PERMISSION_MODE": ("permissions", "mode"),
        }
        for env_key, (section, field) in env_map.items():
            value = os.environ.get(env_key)
            if value is not None:
                setattr(getattr(self, section), field, value)
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_config.py -v`
Expected: All 3 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/config.py tests/test_core/test_config.py configs/default.yaml
git commit -m "feat(core): add YAML config system with env overrides"
```

---

## Task 7: Agent Loop

**Files:**
- Create: `src/norn/core/agent.py`
- Create: `tests/test_core/test_agent.py`

**Step 1: Write failing test**

```python
# tests/test_core/test_agent.py
"""Tests for the core agent loop."""
import pytest
from unittest.mock import AsyncMock, MagicMock

from norn.core.agent import AgentLoop
from norn.core.models import Message, Role, LLMResponse, ToolCall, TokenUsage
from norn.tools.base import RiskLevel, ToolContext, ToolResult
from norn.tools.registry import ToolRegistry
from pydantic import BaseModel


class EchoInput(BaseModel):
    text: str


class EchoTool:
    name = "echo"
    description = "Echo text back"
    risk_level = RiskLevel.LOW
    input_model = EchoInput

    async def execute(self, input: EchoInput, ctx: ToolContext) -> ToolResult:
        return ToolResult(output=f"echo: {input.text}")


@pytest.fixture
def registry():
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


@pytest.fixture
def mock_llm_text_only():
    """LLM that returns text without tool calls."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value=LLMResponse(
        content="Hello, I'm Norn!",
        tool_calls=[],
        usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
    ))
    return llm


@pytest.fixture
def mock_llm_with_tool():
    """LLM that first calls a tool, then responds with text."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        # First call: tool invocation
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="echo", arguments={"text": "test"})],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        ),
        # Second call: final text response
        LLMResponse(
            content="The echo said: test",
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=20, completion_tokens=10, total_tokens=30),
        ),
    ])
    return llm


@pytest.mark.asyncio
async def test_agent_text_response(mock_llm_text_only, registry):
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    result = await agent.run("hello")
    assert result.content == "Hello, I'm Norn!"
    assert mock_llm_text_only.complete.call_count == 1


@pytest.mark.asyncio
async def test_agent_tool_call_then_response(mock_llm_with_tool, registry):
    agent = AgentLoop(llm=mock_llm_with_tool, registry=registry)
    result = await agent.run("echo something")
    assert result.content == "The echo said: test"
    assert mock_llm_with_tool.complete.call_count == 2


@pytest.mark.asyncio
async def test_agent_unknown_tool(mock_llm_text_only, registry):
    """LLM calls a tool that doesn't exist."""
    mock_llm_text_only.complete = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(id="c1", name="nonexistent", arguments={})],
        ),
        LLMResponse(content="Sorry, tool not found.", tool_calls=[]),
    ])
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    result = await agent.run("do something")
    assert result.content == "Sorry, tool not found."


@pytest.mark.asyncio
async def test_agent_history_grows(mock_llm_text_only, registry):
    agent = AgentLoop(llm=mock_llm_text_only, registry=registry)
    await agent.run("first")
    await agent.run("second")
    # History should contain: user1, assistant1, user2, assistant2
    assert len(agent.history) == 4
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_core/test_agent.py -v`
Expected: FAIL

**Step 3: Write implementation**

```python
# src/norn/core/agent.py
"""Core agent loop for Norn."""
from __future__ import annotations

from typing import Any

from norn.core.llm import LLMProvider
from norn.core.models import LLMResponse, Message, Role, ToolCall
from norn.tools.base import ToolContext, ToolResult
from norn.tools.registry import ToolRegistry


class AgentLoop:
    """The main agent loop: message -> LLM -> tool calls -> repeat."""

    MAX_TOOL_ROUNDS = 25  # Safety limit

    def __init__(
        self,
        llm: LLMProvider,
        registry: ToolRegistry,
        system_prompt: str = "You are Norn, a helpful coding agent.",
        cwd: str = ".",
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.ctx = ToolContext(cwd=cwd)
        self.history: list[Message] = []

    async def run(self, user_input: str) -> LLMResponse:
        """Run one turn of the agent loop."""
        self.history.append(Message(role=Role.USER, content=user_input))

        messages = [
            Message(role=Role.SYSTEM, content=self.system_prompt),
            *self.history,
        ]

        for _round in range(self.MAX_TOOL_ROUNDS):
            response = await self.llm.complete(
                messages=messages,
                tools=self.registry.get_schemas() or None,
            )

            if not response.has_tool_calls:
                self.history.append(
                    Message(role=Role.ASSISTANT, content=response.content)
                )
                return response

            # Process tool calls
            assistant_msg = Message(
                role=Role.ASSISTANT,
                content=response.content,
                tool_calls=response.tool_calls,
            )
            messages.append(assistant_msg)

            for call in response.tool_calls:
                result = await self._execute_tool(call)
                tool_msg = Message(
                    role=Role.TOOL,
                    content=result.output or result.error or "",
                    tool_call_id=call.id,
                )
                messages.append(tool_msg)

        # Safety: max rounds reached
        final = LLMResponse(content="[Max tool rounds reached]")
        self.history.append(Message(role=Role.ASSISTANT, content=final.content))
        return final

    async def _execute_tool(self, call: ToolCall) -> ToolResult:
        """Execute a single tool call."""
        tool = self.registry.get(call.name)
        if tool is None:
            return ToolResult(error=f"Unknown tool: {call.name}")

        try:
            input_obj = tool.input_model(**call.arguments)
            return await tool.execute(input_obj, self.ctx)
        except Exception as e:
            return ToolResult(error=f"Tool execution error: {e}")
```

**Step 4: Run test**

Run: `pytest tests/test_core/test_agent.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add src/norn/core/agent.py tests/test_core/test_agent.py
git commit -m "feat(core): add agent loop with tool calling cycle"
```

---

## Task 8: CLI

**Files:**
- Create: `src/norn/cli/main.py`
- Modify: `src/norn/__init__.py`

**Step 1: Write the CLI entrypoint**

```python
# src/norn/cli/main.py
"""Norn CLI entrypoint."""
from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.prompt import Prompt

from norn.core.agent import AgentLoop
from norn.core.config import NornConfig
from norn.core.llm import LiteLLMProvider
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry

app = typer.Typer(name="norn", help="Norn - the coding agent that weaves your destiny")
console = Console()


def _build_registry() -> ToolRegistry:
    """Build the default tool registry."""
    registry = ToolRegistry()
    registry.register(BashTool())
    registry.register(FileReadTool())
    registry.register(FileWriteTool())
    registry.register(FileEditTool())
    registry.register(GlobTool())
    registry.register(GrepTool())
    return registry


def _build_provider(config: NornConfig) -> LiteLLMProvider:
    """Build the LLM provider from config."""
    model = config.llm.model
    if config.llm.provider == "ollama":
        model = f"ollama/{config.llm.model}"
    elif config.llm.provider == "openrouter":
        model = f"openrouter/{config.llm.model}"
    return LiteLLMProvider(model=model, api_base=config.llm.api_base)


@app.command()
def chat() -> None:
    """Start an interactive chat session."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
    )

    console.print("[bold]Norn[/bold] - the coding agent that weaves your destiny")
    console.print("Type 'exit' or 'quit' to leave. Ctrl+C to interrupt.\n")

    async def _chat_loop() -> None:
        while True:
            try:
                user_input = Prompt.ask("[bold cyan]>[/bold cyan]")
            except (EOFError, KeyboardInterrupt):
                console.print("\nGoodbye.")
                break

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("Goodbye.")
                break

            if not user_input.strip():
                continue

            try:
                with console.status("[dim]Thinking...[/dim]"):
                    response = await agent.run(user_input)

                if response.content:
                    console.print(Markdown(response.content))
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Interrupted.[/dim]")
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")

    asyncio.run(_chat_loop())


@app.command()
def run(prompt: str = typer.Argument(help="One-shot prompt to execute")) -> None:
    """Run a one-shot prompt and exit."""
    config = NornConfig.load()
    config.apply_env_overrides()

    provider = _build_provider(config)
    registry = _build_registry()
    agent = AgentLoop(
        llm=provider,
        registry=registry,
        cwd=str(Path.cwd()),
    )

    async def _run_once() -> None:
        response = await agent.run(prompt)
        if response.content:
            console.print(Markdown(response.content))

    asyncio.run(_run_once())


@app.command()
def tools() -> None:
    """List available tools."""
    registry = _build_registry()
    console.print("[bold]Available tools:[/bold]\n")
    for tool in registry.list_tools():
        risk_color = {"low": "green", "medium": "yellow", "high": "red"}[tool.risk_level.value]
        console.print(
            f"  [{risk_color}]{tool.risk_level.value:>6}[/{risk_color}]  "
            f"[bold]{tool.name}[/bold] - {tool.description}"
        )


@app.command()
def config() -> None:
    """Show current configuration."""
    cfg = NornConfig.load()
    cfg.apply_env_overrides()
    console.print("[bold]Current configuration:[/bold]\n")
    console.print(f"  LLM provider: {cfg.llm.provider}")
    console.print(f"  LLM model:    {cfg.llm.model}")
    console.print(f"  Permissions:  {cfg.permissions.mode.value}")
    console.print(f"  Dream:        {'enabled' if cfg.flags.dream_system else 'disabled'}")
    console.print(f"  Coordinator:  {'enabled' if cfg.flags.coordinator else 'disabled'}")


@app.command()
def version() -> None:
    """Show Norn version."""
    console.print("norn 0.1.0")


if __name__ == "__main__":
    app()
```

**Step 2: Test CLI manually**

Run:
```bash
norn version
norn tools
norn config
```
Expected: Version, tool list, and config displayed

**Step 3: Commit**

```bash
git add src/norn/cli/main.py
git commit -m "feat(cli): add interactive chat, one-shot run, tools, and config commands"
```

---

## Task 9: Integration Test

**Files:**
- Create: `tests/test_integration.py`

**Step 1: Write integration test**

```python
# tests/test_integration.py
"""Integration test: full agent loop with mocked LLM."""
import pytest
from unittest.mock import AsyncMock

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse, ToolCall, TokenUsage
from norn.tools.bash_tool import BashTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
from norn.tools.file_edit import FileEditTool
from norn.tools.glob_tool import GlobTool
from norn.tools.grep_tool import GrepTool
from norn.tools.registry import ToolRegistry


@pytest.fixture
def full_registry():
    reg = ToolRegistry()
    reg.register(BashTool())
    reg.register(FileReadTool())
    reg.register(FileWriteTool())
    reg.register(FileEditTool())
    reg.register(GlobTool())
    reg.register(GrepTool())
    return reg


@pytest.mark.asyncio
async def test_full_loop_write_then_read(full_registry, tmp_path):
    """LLM writes a file, then reads it back."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        # Step 1: Write file
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="c1",
                name="file_write",
                arguments={"path": str(tmp_path / "test.py"), "content": "print('hello')"},
            )],
        ),
        # Step 2: Read file back
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="c2",
                name="file_read",
                arguments={"path": str(tmp_path / "test.py")},
            )],
        ),
        # Step 3: Final response
        LLMResponse(content="Done! File written and verified.", tool_calls=[]),
    ])

    agent = AgentLoop(llm=llm, registry=full_registry, cwd=str(tmp_path))
    result = await agent.run("write a hello world script and verify it")

    assert result.content == "Done! File written and verified."
    assert (tmp_path / "test.py").read_text() == "print('hello')"
    assert llm.complete.call_count == 3


@pytest.mark.asyncio
async def test_full_loop_bash_execution(full_registry, tmp_path):
    """LLM runs a bash command."""
    llm = AsyncMock()
    llm.complete = AsyncMock(side_effect=[
        LLMResponse(
            content=None,
            tool_calls=[ToolCall(
                id="c1",
                name="bash",
                arguments={"command": "echo 'norn is alive'"},
            )],
        ),
        LLMResponse(content="Command executed successfully.", tool_calls=[]),
    ])

    agent = AgentLoop(llm=llm, registry=full_registry, cwd=str(tmp_path))
    result = await agent.run("run echo")
    assert result.content == "Command executed successfully."


@pytest.mark.asyncio
async def test_all_tools_registered(full_registry):
    """Verify all 6 Phase 1 tools are registered."""
    expected = {"bash", "file_read", "file_write", "file_edit", "glob", "grep"}
    actual = {t.name for t in full_registry.list_tools()}
    assert actual == expected


@pytest.mark.asyncio
async def test_schemas_valid_for_llm(full_registry):
    """Verify schemas are valid for LLM consumption."""
    schemas = full_registry.get_schemas()
    assert len(schemas) == 6
    for schema in schemas:
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
```

**Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: All 4 tests PASS

**Step 3: Run full test suite**

Run: `pytest tests/ -v --tb=short`
Expected: All tests PASS

**Step 4: Run quality checks**

Run: `ruff check src/ tests/`
Expected: No issues (or only minor fixable ones)

**Step 5: Final commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration tests for full agent loop"
```

---

## Verification Checklist

After all tasks are done, verify:

- [ ] `uv pip install -e ".[dev]"` succeeds
- [ ] `pytest tests/ -v` — all tests pass
- [ ] `ruff check src/ tests/` — no lint errors
- [ ] `norn version` — prints version
- [ ] `norn tools` — lists 6 tools
- [ ] `norn config` — shows configuration
- [ ] `norn chat` — starts interactive session (requires running LLM)
- [ ] `git log --oneline` — shows 9 clean commits
