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
        # W1.3: store the last system_prompt so _llm_summarize shares the same
        # prefix as the main agent loop → Anthropic prompt-cache hits on the prefix.
        self._system_prompt: str = ""

    async def build_messages(
        self,
        system_prompt: str,
        history: list[Message],
    ) -> list[Message]:
        """Build message list with optional sliding window.

        Returns: [system, summary_msg?, ...recent_turns]
        """
        self._system_prompt = system_prompt  # W1.3: shared prefix for summarization
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
        """Use an LLM to generate the summary.

        Prepends the agent's system prompt so the API call shares the same prefix
        as the main loop → prompt-cache hits on the system message tokens.
        """
        conversation = "\n".join(f"{m.role.value}: {(m.content or '')[:200]}" for m in messages)
        prompt = _SUMMARY_PROMPT.format(
            max_tokens=self.summary_max_tokens,
            conversation=conversation,
        )
        call_messages: list[Message] = []
        if self._system_prompt:
            call_messages.append(Message(role=Role.SYSTEM, content=self._system_prompt))
        call_messages.append(Message(role=Role.USER, content=prompt))
        response = await self._summary_provider.complete(  # type: ignore[union-attr]
            messages=call_messages,
            tools=None,
            max_tokens=self.summary_max_tokens,
        )
        return response.content or ""

    def _extractive_summary(self, messages: list[Message]) -> str:
        """Extractive summary with tool call collapse.

        Collapses tool_call + tool_result pairs into brief one-liners.
        Preserves user questions and key assistant responses.
        """
        parts: list[str] = []
        i = 0
        while i < len(messages) and len(parts) < 12:
            msg = messages[i]

            if msg.role == Role.USER and msg.content:
                parts.append(f"- User: {msg.content[:100]}")
            elif msg.role == Role.ASSISTANT:
                if msg.tool_calls:
                    # Collapse tool calls: show tool name + brief args + result preview
                    for tc in msg.tool_calls:
                        args_summary = self._summarize_tool_args(tc)
                        # Look ahead for the matching tool result
                        result_preview = ""
                        for j in range(i + 1, min(i + 1 + len(msg.tool_calls) + 2, len(messages))):
                            if messages[j].role == Role.TOOL and messages[j].tool_call_id == tc.id:
                                result_content = messages[j].content or ""
                                result_preview = result_content[:60].replace("\n", " ").strip()
                                break
                        parts.append(f"- Tool[{tc.name}]: {args_summary} -> {result_preview}")
                elif msg.content and len(msg.content) > 20:
                    parts.append(f"- Assistant: {msg.content[:80]}")
            # Skip standalone TOOL messages (handled via look-ahead above)
            i += 1

        return "\n".join(parts) if parts else "No prior context."

    @staticmethod
    def _summarize_tool_args(tool_call) -> str:
        """Produce a brief summary of tool call arguments."""
        args = tool_call.arguments
        if "command" in args:
            cmd = str(args["command"])
            return cmd[:40] + ("..." if len(cmd) > 40 else "")
        if "path" in args:
            return str(args["path"])
        if "file_path" in args:
            return str(args["file_path"])
        if "pattern" in args:
            return str(args["pattern"])
        for v in args.values():
            if isinstance(v, str):
                return v[:40]
        return "..."

    def _estimate_tokens(self, messages: list[Message]) -> int:
        """Estimate token count for a list of messages.

        Uses chars/4 heuristic. For production, integrate litellm.token_counter().
        """
        total_chars = sum(len(m.content or "") for m in messages)
        return total_chars // 4
