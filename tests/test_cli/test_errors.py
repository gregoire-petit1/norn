"""Tests for user-friendly error formatting."""

from __future__ import annotations

from norn.cli.errors import format_llm_error


def test_format_429_error():
    """429 errors should produce a user-friendly message."""
    err = Exception(
        'OllamaException - {"StatusCode":429,"Status":"429 Too Many Requests",'
        '"error":"you have reached your session usage limit"}'
    )
    msg = format_llm_error(err)
    assert "rate limit" in msg.lower() or "Rate limit" in msg


def test_format_too_many_requests():
    """'Too Many Requests' should be recognized."""
    err = Exception("Too Many Requests")
    msg = format_llm_error(err)
    assert "rate limit" in msg.lower() or "Rate limit" in msg


def test_format_connection_refused():
    """Connection refused should produce a user-friendly message."""
    err = ConnectionError("Connection refused")
    msg = format_llm_error(err)
    assert "connect" in msg.lower() or "Connect" in msg


def test_format_timeout():
    """Timeout errors should produce a user-friendly message."""
    err = Exception("Request timeout after 30s")
    msg = format_llm_error(err)
    assert "timeout" in msg.lower() or "timed out" in msg.lower()


def test_format_generic_error():
    """Unknown errors should include the original message."""
    err = Exception("something weird happened")
    msg = format_llm_error(err)
    assert "something weird" in msg


def test_format_ollama_session_limit_inside_api_connection_error():
    """Ollama cloud surfaces session caps as APIConnectionError. The body
    contains the real reason — we must report it as a rate limit rather
    than "cannot connect" (which sends users to debug the wrong layer)."""
    err = Exception(
        'litellm.APIConnectionError: OllamaException - {"error":"you '
        '(gregoire_petit) have reached your session usage limit, '
        'upgrade for higher limits: https://ollama.com/upgrade"}'
    )
    msg = format_llm_error(err)
    assert "rate limit" in msg.lower()
    assert "cannot connect" not in msg.lower()
