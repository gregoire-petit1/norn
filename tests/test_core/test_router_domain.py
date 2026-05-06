"""Tests for domain-aware routing signals (Phase 10)."""

from __future__ import annotations

from norn.core.models import Message, Role
from norn.core.router import _score_complexity


def _msg(content: str) -> list[Message]:
    return [Message(role=Role.USER, content=content)]


class TestDomainRouting:
    """Test domain-aware routing signals."""

    def test_architecture_domain_boosts_score(self):
        msgs = _msg("Design a microservice architecture for the payment system")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True
        assert score >= 2

    def test_security_domain_boosts_score(self):
        msgs = _msg("Check for SQL injection vulnerabilities in the auth module")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True
        assert score >= 2

    def test_simple_question_reduces_score(self):
        msgs = _msg("What is the Python version?")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("simple_pattern", False) is True
        assert score <= 1

    def test_ml_training_boosts_score(self):
        msgs = _msg("Fix the training loop gradient accumulation bug")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True

    def test_no_domain_match_unchanged(self):
        msgs = _msg("Add a print statement to line 42")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is False
        assert signals.get("simple_pattern", False) is False

    def test_debugging_domain_boosts(self):
        msgs = _msg("There's a race condition in the connection pool")
        score, signals = _score_complexity(msgs, [])
        assert signals.get("domain_boost", False) is True

    def test_domain_routing_disabled(self):
        msgs = _msg("Design a microservice architecture")
        score, signals = _score_complexity(msgs, [], domain_routing=False)
        assert "domain_boost" not in signals or signals.get("domain_boost") is False
        assert "simple_pattern" not in signals or signals.get("simple_pattern") is False
