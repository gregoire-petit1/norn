"""Tests for LLM provider abstraction."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norn.core.llm import LiteLLMProvider, build_tool_schemas
from norn.core.models import LLMResponse, Message, Role


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
        result = await provider.complete(messages=[Message(role=Role.USER, content="Hi")])
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
            tools=[
                {
                    "name": "bash",
                    "description": "Run shell",
                    "parameters": {"type": "object", "properties": {"command": {"type": "string"}}},
                }
            ],
        )
        assert result.has_tool_calls is True
        assert result.tool_calls[0].name == "bash"
        assert result.tool_calls[0].arguments == {"command": "ls"}
