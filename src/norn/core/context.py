"""Sliding window context manager with history summarization.

Manages conversation history to keep total context within a token budget.
Older turns are evicted and replaced with a condensed summary. Recent
turns are always preserved verbatim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from norn.core.models import Message, Role

if TYPE_CHECKING:
    from norn.core.llm import LLMProvider

# Default summarization prompt
_SUMMARY_PROMPT = (
    "Summarize the conversation below in {max_tokens} tokens or less.\n"
    "Preserve: key decisions, file paths mentioned, task context, errors encountered.\n"
    "Omit: verbose tool outputs, intermediate steps that led nowhere.\n"
    "Format: dense bullet points.\n\n"
    "{conversation}"
)


class ContextManager:
    """Manages conversation history with sliding window and summarization.

    When total history exceeds max_history_tokens, older messages are
    evicted and replaced with a summary. The last `recent_turns_keep`
    turns are always preserved verbatim.
    """

    def __init__(
        self,
        max_history_tokens: int = 8000,
        recent_turns_keep: int = 6,
        summary_max_tokens: int = 300,
        summary_provider: LLMProvider | None = None,
        enabled: bool = True,
    ) -> None:
        self.max_history_tokens = max_history_tokens
        self.recent_turns_keep = recent_turns_keep
        self.summary_max_tokens = summary_max_tokens
        self._summary_provider = summary_provider
        self.enabled = enabled
        # Cache
        self._cached_summary: str | None = None
        self._cached_history_len: int = 0

    async def build_messages(
        self,
        system_prompt: str,
        history: list[Message],
    ) -> list[Message]:
        """Build message list with optional sliding window.

        Returns: [system, summary_msg?, ...recent_turns]
        """
        system_msg = Message(role=Role.SYSTEM, content=system_prompt)

        if not self.enabled or not history:
            return [system_msg, *history]

        total_tokens = self._estimate_tokens(history)
        if total_tokens <= self.max_history_tokens:
            return [system_msg, *history]

        # Need to window: keep last N turns (each turn = user + assistant = 2 msgs)
        recent_msg_count = self.recent_turns_keep * 2
        recent_msg_count = min(recent_msg_count, len(history))

        recent = history[-recent_msg_count:] if recent_msg_count > 0 else []
        old = history[:-recent_msg_count] if recent_msg_count < len(history) else []

        # Generate or use cached summary
        summary = await self._get_summary(old, len(history))

        messages = [system_msg]
        if summary:
            messages.append(
                Message(role=Role.SYSTEM, content=f"## Conversation Summary\n\n{summary}")
            )
        messages.extend(recent)
        return messages

    async def _get_summary(self, old_messages: list[Message], history_len: int) -> str | None:
        """Get summary of old messages, using cache when possible."""
        if not old_messages:
            return None

        # Use cache if history hasn't grown
        if self._cached_summary and self._cached_history_len == history_len:
            return self._cached_summary

        summary = await self._summarize(old_messages)
        self._cached_summary = summary
        self._cached_history_len = history_len
        return summary

    async def _summarize(self, messages: list[Message]) -> str:
        """Generate a condensed summary of messages.

        If no summary_provider is set, uses a simple extractive fallback.
        """
        if self._summary_provider is not None:
            return await self._llm_summarize(messages)
        return self._extractive_summary(messages)

    async def _llm_summarize(self, messages: list[Message]) -> str:
        """Use an LLM to generate the summary."""
        conversation = "\n".join(f"{m.role.value}: {(m.content or '')[:200]}" for m in messages)
        prompt = _SUMMARY_PROMPT.format(
            max_tokens=self.summary_max_tokens,
            conversation=conversation,
        )
        response = await self._summary_provider.complete(  # type: ignore[union-attr]
            messages=[Message(role=Role.USER, content=prompt)],
            tools=None,
            max_tokens=self.summary_max_tokens,
        )
        return response.content or ""

    def _extractive_summary(self, messages: list[Message]) -> str:
        """Simple extractive summary when no LLM is available.

        Keeps first user message + key assistant messages (non-tool).
        """
        parts: list[str] = []
        for msg in messages:
            if msg.role == Role.USER and msg.content:
                parts.append(f"- User: {msg.content[:100]}")
            elif msg.role == Role.ASSISTANT and msg.content and len(msg.content) > 20:
                parts.append(f"- Assistant: {msg.content[:80]}")
            if len(parts) >= 10:
                break
        return "\n".join(parts) if parts else "No prior context."

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.

        Uses chars/4 heuristic. For production, integrate litellm.token_counter().
        """
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // 4
