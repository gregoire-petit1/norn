"""Tests for the coordinator activation heuristic."""

from __future__ import annotations

from norn.coordinator.heuristic import ActivationHeuristic


class TestIndicators:
    def test_long_text_detected(self):
        """Task text > 500 chars is an indicator."""
        h = ActivationHeuristic()
        text = "x" * 501
        result = h.evaluate(text)
        assert "long_text" in result.indicators

    def test_short_text_not_detected(self):
        h = ActivationHeuristic()
        result = h.evaluate("Fix the typo in README")
        assert "long_text" not in result.indicators

    def test_keywords_detected(self):
        """Keywords like 'refactor', 'across', 'multiple files' trigger."""
        h = ActivationHeuristic()
        result = h.evaluate("Refactor the auth module across multiple files")
        assert "keywords" in result.indicators

    def test_no_keywords(self):
        h = ActivationHeuristic()
        result = h.evaluate("Fix the typo")
        assert "keywords" not in result.indicators

    def test_file_count_detected(self):
        """Mentioning many files triggers the indicator."""
        h = ActivationHeuristic()
        text = "Update these files: " + ", ".join(f"src/mod{i}.py" for i in range(12))
        result = h.evaluate(text)
        assert "many_files" in result.indicators

    def test_few_files_not_detected(self):
        h = ActivationHeuristic()
        result = h.evaluate("Update src/main.py and src/utils.py")
        assert "many_files" not in result.indicators

    def test_explicit_parallel_keyword(self):
        """Words like 'parallel', 'concurrent', 'simultaneously' trigger."""
        h = ActivationHeuristic()
        result = h.evaluate("Run these checks in parallel across the codebase")
        assert "parallel_request" in result.indicators


class TestActivation:
    def test_activates_with_two_indicators(self):
        """Coordinator activates when >= 2 indicators are present."""
        h = ActivationHeuristic(threshold=2)
        # Long text + keywords
        text = "x" * 501 + " refactor across multiple modules"
        result = h.evaluate(text)
        assert result.should_activate is True

    def test_does_not_activate_with_one(self):
        """Single indicator is not enough."""
        h = ActivationHeuristic(threshold=2)
        result = h.evaluate("x" * 501)  # Only long text
        assert result.should_activate is False

    def test_does_not_activate_with_zero(self):
        h = ActivationHeuristic(threshold=2)
        result = h.evaluate("Fix typo")
        assert result.should_activate is False

    def test_custom_threshold(self):
        """Custom threshold works."""
        h = ActivationHeuristic(threshold=1)
        result = h.evaluate("Refactor the module")
        assert result.should_activate is True  # keywords alone is enough

    def test_result_includes_reason(self):
        h = ActivationHeuristic(threshold=2)
        text = "x" * 501 + " refactor across multiple files"
        result = h.evaluate(text)
        assert result.reason is not None
        assert len(result.reason) > 0
