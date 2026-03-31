"""Dream system prompts for memory consolidation."""

from __future__ import annotations


def build_dream_system_prompt() -> str:
    """Build the system prompt for the dream consolidation LLM call."""
    return """\
You are Norn's Dream Engine -- a memory consolidation system.

Your job is to process recent session logs and update the persistent memory files.
You are maintaining a knowledge base about the user, their projects, preferences, and ongoing work.

## Rules

1. **MEMORY.md** is the main index. Keep it under 200 lines and ~25KB.
2. Use **absolute dates** (2026-03-31), never relative dates (yesterday, last week).
3. **Delete contradicted facts** -- if new info contradicts old info, keep only the new.
4. **Merge duplicates** -- don't repeat the same fact in multiple places.
5. **Be concise** -- bullet points, not paragraphs.
6. **Preserve structure** -- maintain headers and sections in MEMORY.md.
7. **Topic files** -- create/update topic files for major ongoing projects or themes.
8. **Prune stale info** -- remove facts that are clearly outdated (> 30 days with no references).

## Output Format

Return a JSON object with these fields:
- `memory`: string -- the updated MEMORY.md content
- `topics`: dict[str, str] -- topic name -> content (only include topics that changed)
- `pruned_topics`: list[str] -- topic names to delete (stale/empty)
- `summary`: string -- one-line description of what changed
"""


def build_consolidation_prompt(
    current_memory: str,
    daily_logs: dict[str, str],
    topic_files: dict[str, str],
) -> str:
    """Build the user prompt with all context for consolidation."""
    parts: list[str] = []

    # Current memory
    if current_memory.strip():
        parts.append("## Current MEMORY.md\n")
        parts.append(f"```markdown\n{current_memory}\n```\n")
    else:
        parts.append("## Current MEMORY.md\n")
        parts.append("(Empty -- no existing memory yet)\n")

    # Daily logs (most recent first)
    parts.append("## Recent Daily Logs\n")
    if daily_logs:
        for date in sorted(daily_logs.keys(), reverse=True):
            parts.append(f"### {date}\n")
            parts.append(f"{daily_logs[date]}\n")
    else:
        parts.append("(No daily logs to process)\n")

    # Topic files
    parts.append("## Existing Topic Files\n")
    if topic_files:
        for name, content in sorted(topic_files.items()):
            parts.append(f"### topics/{name}.md\n")
            parts.append(f"```markdown\n{content}\n```\n")
    else:
        parts.append("(No topic files yet)\n")

    parts.append(
        "\nPlease consolidate the above into updated memory. "
        "Return JSON as specified in the system prompt."
    )

    return "\n".join(parts)
