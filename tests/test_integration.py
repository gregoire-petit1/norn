"""Integration test: full agent loop with mocked LLM."""

from unittest.mock import AsyncMock

import pytest

from norn.core.agent import AgentLoop
from norn.core.models import LLMResponse, ToolCall
from norn.tools.bash_tool import BashTool
from norn.tools.file_edit import FileEditTool
from norn.tools.file_read import FileReadTool
from norn.tools.file_write import FileWriteTool
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
    llm.complete = AsyncMock(
        side_effect=[
            # Step 1: Write file
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="file_write",
                        arguments={"path": str(tmp_path / "test.py"), "content": "print('hello')"},
                    )
                ],
            ),
            # Step 2: Read file back
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c2",
                        name="file_read",
                        arguments={"path": str(tmp_path / "test.py")},
                    )
                ],
            ),
            # Step 3: Final response
            LLMResponse(content="Done! File written and verified.", tool_calls=[]),
        ]
    )

    agent = AgentLoop(llm=llm, registry=full_registry, cwd=str(tmp_path))
    result = await agent.run("write a hello world script and verify it")

    assert result.content == "Done! File written and verified."
    assert (tmp_path / "test.py").read_text() == "print('hello')"
    assert llm.complete.call_count == 3


@pytest.mark.asyncio
async def test_full_loop_bash_execution(full_registry, tmp_path):
    """LLM runs a bash command."""
    llm = AsyncMock()
    llm.complete = AsyncMock(
        side_effect=[
            LLMResponse(
                content=None,
                tool_calls=[
                    ToolCall(
                        id="c1",
                        name="bash",
                        arguments={"command": "echo 'norn is alive'"},
                    )
                ],
            ),
            LLMResponse(content="Command executed successfully.", tool_calls=[]),
        ]
    )

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
