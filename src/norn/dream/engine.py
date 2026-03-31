"""Dream engine: four-phase memory consolidation."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from norn.core.models import Message, Role
from norn.core.utils import extract_json
from norn.dream.prompts import build_consolidation_prompt, build_dream_system_prompt

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider
    from norn.memory.store import MemoryStore

logger = logging.getLogger(__name__)


@dataclass
class DreamResult:
    """Result of a dream consolidation."""

    success: bool
    summary: str | None = None
    error: str | None = None
    files_written: list[str] = field(default_factory=list)
    files_pruned: list[str] = field(default_factory=list)


class DreamEngine:
    """Orchestrates the four-phase memory consolidation."""

    def __init__(self, store: MemoryStore, llm: LLMProvider) -> None:
        self._store = store
        self._llm = llm

    # --- Phase 1: Orient ---

    def _orient(self) -> dict[str, Any]:
        """Read current memory state."""
        return {
            "current_memory": self._store.read_memory(),
            "topic_files": {
                name: self._store.read_topic(name) or "" for name in self._store.list_topics()
            },
        }

    # --- Phase 2: Gather ---

    def _gather(self) -> dict[str, str]:
        """Collect recent daily logs."""
        logs: dict[str, str] = {}
        for date_str in self._store.list_daily_logs():
            content = self._store.read_daily(date_str)
            if content:
                logs[date_str] = content
        return logs

    # --- Phase 3: Consolidate ---

    async def _consolidate(
        self,
        current_memory: str,
        daily_logs: dict[str, str],
        topic_files: dict[str, str],
    ) -> dict[str, Any]:
        """Send context to LLM and get consolidated memory."""
        system_prompt = build_dream_system_prompt()
        user_prompt = build_consolidation_prompt(
            current_memory=current_memory,
            daily_logs=daily_logs,
            topic_files=topic_files,
        )

        messages = [
            Message(role=Role.SYSTEM, content=system_prompt),
            Message(role=Role.USER, content=user_prompt),
        ]

        response = await self._llm.complete(messages=messages, tools=None)
        content = response.content or ""

        return json.loads(extract_json(content))

    # --- Phase 4: Prune ---

    def _prune(self, consolidation: dict[str, Any]) -> tuple[list[str], list[str]]:
        """Apply consolidation results to memory files."""
        files_written: list[str] = []
        files_pruned: list[str] = []

        # Update MEMORY.md
        memory_content = consolidation.get("memory", "")
        if memory_content:
            self._store.write_memory(memory_content)
            files_written.append("MEMORY.md")

        # Update topic files
        topics = consolidation.get("topics", {})
        for name, content in topics.items():
            self._store.write_topic(name, content)
            files_written.append(f"topics/{name}.md")

        # Delete pruned topics
        pruned = consolidation.get("pruned_topics", [])
        for name in pruned:
            if self._store.delete_topic(name):
                files_pruned.append(f"topics/{name}.md")

        return files_written, files_pruned

    # --- Full dream cycle ---

    async def dream(self) -> DreamResult:
        """Run the complete four-phase dream consolidation."""
        try:
            # Phase 1: Orient
            context = self._orient()

            # Phase 2: Gather
            daily_logs = self._gather()

            # Phase 3: Consolidate
            consolidation = await self._consolidate(
                current_memory=context["current_memory"],
                daily_logs=daily_logs,
                topic_files=context["topic_files"],
            )

            # Phase 4: Prune
            files_written, files_pruned = self._prune(consolidation)

            summary = consolidation.get("summary", "Memory consolidated.")
            return DreamResult(
                success=True,
                summary=summary,
                files_written=files_written,
                files_pruned=files_pruned,
            )

        except json.JSONDecodeError as e:
            logger.error("Dream failed: invalid JSON from LLM: %s", e)
            return DreamResult(success=False, error=f"Invalid JSON from LLM: {e}")
        except Exception as e:
            logger.error("Dream failed: %s", e)
            return DreamResult(success=False, error=str(e))
