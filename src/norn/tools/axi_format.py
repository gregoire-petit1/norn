"""AXI-style tool-output formatting (SOTA v2, wave 2 — B).

Helpers that put a token-efficient summary line FIRST (aggregates, truncation
signal, next-step hint), so an agent learns counts and whether output was cut
without re-parsing the body or re-issuing the call. The lead line is emitted
first on purpose: it survives head-retention truncation (core/truncation.py).

Pure functions, no imports from tools — safe to import anywhere. Opt-in per
tool behind the `axi_output` flag; the default (flag off) path is unchanged.
"""

from __future__ import annotations


def axi_empty(noun: str, next_step: str) -> str:
    """Explicit empty state with a concrete next step."""
    return f"No {noun}. Next: {next_step}"


def axi_match_result(
    lines: list[str],
    *,
    total_matches: int,
    file_count: int,
    truncated: bool,
    full_hint: str,
) -> str:
    """Match listing (grep) with a count/file/truncation lead line."""
    if not lines:
        return axi_empty("matches", full_hint)
    m_plural = "es" if total_matches != 1 else ""
    f_plural = "s" if file_count != 1 else ""
    head = f"{total_matches} match{m_plural} in {file_count} file{f_plural}"
    if truncated:
        head += f" (showing {len(lines)}, truncated — narrow with {full_hint})"
    return head + "\n" + "\n".join(lines)


def axi_list_result(
    items: list[str],
    *,
    total: int,
    shown: int,
    noun: str,
    full_hint: str,
    empty_hint: str,
) -> str:
    """Generic listing (glob) with a count/truncation lead line."""
    if not items:
        return axi_empty(noun, empty_hint)
    head = f"{total} {noun}"
    if shown < total:
        head += f" (showing {shown} — narrow with {full_hint})"
    return head + "\n" + "\n".join(items)


def axi_file_header(path: str, *, start: int, end: int, total_lines: int) -> str:
    """Header for a file slice so the agent never re-reads to learn length."""
    return f"{path}: lines {start}-{end} of {total_lines}"


def axi_dir_header(*, dirs: int, files: int) -> str:
    """Header for a directory listing with entry counts."""
    total = dirs + files
    return f"{total} entr{'ies' if total != 1 else 'y'} ({dirs} dirs, {files} files)"
