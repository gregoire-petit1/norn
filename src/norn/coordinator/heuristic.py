"""Activation heuristic: decides when coordinator mode is justified."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_COORDINATOR_KEYWORDS = [
    r"\brefactor\b",
    r"\bacross\b",
    r"\bmultiple files\b",
    r"\bmulti-file\b",
    r"\bcodebase\b",
    r"\bmodules?\b.*\b(update|change|modify|rewrite)\b",
    r"\b(update|change|modify|rewrite)\b.*\bmodules?\b",
    r"\bsystem-wide\b",
    r"\bproject-wide\b",
]

_PARALLEL_KEYWORDS = [
    r"\bparallel\b",
    r"\bconcurrent(ly)?\b",
    r"\bsimultaneous(ly)?\b",
    r"\bin parallel\b",
]

# Matches paths like src/foo.py, tests/bar.ts, etc.
_FILE_PATTERN = re.compile(r"\b[\w/\\.-]+\.\w{1,5}\b")


@dataclass
class ActivationResult:
    """Result of the activation heuristic evaluation."""

    should_activate: bool
    indicators: list[str] = field(default_factory=list)
    reason: str = ""


class ActivationHeuristic:
    """Evaluates whether a task warrants coordinator mode."""

    def __init__(
        self,
        threshold: int = 2,
        min_text_length: int = 500,
        min_file_mentions: int = 10,
    ) -> None:
        self._threshold = threshold
        self._min_text_length = min_text_length
        self._min_file_mentions = min_file_mentions

    def evaluate(self, task_text: str) -> ActivationResult:
        """Evaluate whether coordinator mode should activate."""
        indicators: list[str] = []

        # Indicator 1: Long text
        if len(task_text) > self._min_text_length:
            indicators.append("long_text")

        # Indicator 2: Coordinator keywords
        text_lower = task_text.lower()
        if any(re.search(kw, text_lower) for kw in _COORDINATOR_KEYWORDS):
            indicators.append("keywords")

        # Indicator 3: Many file mentions
        file_matches = _FILE_PATTERN.findall(task_text)
        if len(file_matches) >= self._min_file_mentions:
            indicators.append("many_files")

        # Indicator 4: Parallel request
        if any(re.search(kw, text_lower) for kw in _PARALLEL_KEYWORDS):
            indicators.append("parallel_request")

        should_activate = len(indicators) >= self._threshold
        reason = (
            f"Activated: {len(indicators)}/{self._threshold} indicators ({', '.join(indicators)})"
            if should_activate
            else f"Not activated: {len(indicators)}/{self._threshold} indicators"
        )

        return ActivationResult(
            should_activate=should_activate,
            indicators=indicators,
            reason=reason,
        )
