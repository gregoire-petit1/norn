"""Tests for the image_read multimodal tool."""

from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from norn.tools.base import ToolContext, ToolErrorType
from norn.tools.vision.image_read import ImageReadInput, ImageReadTool

# 1x1 transparent PNG
_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
)


@pytest.fixture
def tool():
    return ImageReadTool(model="github_copilot/claude-sonnet-4.5")


def test_metadata(tool):
    assert tool.name == "image_read"
    assert "image" in tool.description.lower()


@pytest.mark.asyncio
async def test_missing_file_returns_not_found(tool, tmp_path):
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ImageReadInput(file_path="nope.png", instruction="x"), ctx)
    assert result.is_error
    assert result.error_type == ToolErrorType.FILE_NOT_FOUND.value


@pytest.mark.asyncio
async def test_unsupported_format(tool, tmp_path):
    (tmp_path / "f.bmp").write_bytes(b"x")
    ctx = ToolContext(cwd=str(tmp_path))
    result = await tool.execute(ImageReadInput(file_path="f.bmp", instruction="x"), ctx)
    assert result.is_error
    assert result.error_type == ToolErrorType.NOT_SUPPORTED.value


@pytest.mark.asyncio
async def test_successful_read_returns_description(tool, tmp_path):
    (tmp_path / "board.png").write_bytes(_PNG_BYTES)
    ctx = ToolContext(cwd=str(tmp_path))

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(content="A white pixel."))]

    with patch("litellm.acompletion", new=AsyncMock(return_value=fake_response)) as mock_llm:
        result = await tool.execute(
            ImageReadInput(file_path="board.png", instruction="describe"), ctx
        )

    assert not result.is_error
    assert "white pixel" in result.output.lower()
    # Verify a multimodal message with an image_url block was sent
    sent = mock_llm.call_args.kwargs["messages"][0]["content"]
    types = [part["type"] for part in sent]
    assert "image_url" in types
    assert "text" in types
    assert sent[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_api_base_forwarded(tmp_path):
    tool = ImageReadTool(model="m", api_base="http://localhost:1234")
    (tmp_path / "x.png").write_bytes(_PNG_BYTES)
    ctx = ToolContext(cwd=str(tmp_path))
    fake = MagicMock()
    fake.choices = [MagicMock(message=MagicMock(content="ok"))]
    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)) as mock_llm:
        await tool.execute(ImageReadInput(file_path="x.png", instruction="d"), ctx)
    assert mock_llm.call_args.kwargs["api_base"] == "http://localhost:1234"


@pytest.mark.asyncio
async def test_vision_call_failure_is_wrapped(tool, tmp_path):
    (tmp_path / "x.png").write_bytes(_PNG_BYTES)
    ctx = ToolContext(cwd=str(tmp_path))
    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("boom"))):
        result = await tool.execute(ImageReadInput(file_path="x.png", instruction="d"), ctx)
    assert result.is_error
    assert "boom" in result.error


@pytest.mark.asyncio
async def test_empty_description_is_error(tool, tmp_path):
    (tmp_path / "x.png").write_bytes(_PNG_BYTES)
    ctx = ToolContext(cwd=str(tmp_path))
    fake = MagicMock()
    fake.choices = [MagicMock(message=MagicMock(content="   "))]
    with patch("litellm.acompletion", new=AsyncMock(return_value=fake)):
        result = await tool.execute(ImageReadInput(file_path="x.png", instruction="d"), ctx)
    assert result.is_error
