"""Tests for AXI output helpers (wave 2 — B)."""

from norn.tools.axi_format import (
    axi_dir_header,
    axi_empty,
    axi_file_header,
    axi_list_result,
    axi_match_result,
)


def test_empty_has_next_step():
    assert axi_empty("matches", "broaden the pattern") == "No matches. Next: broaden the pattern"


def test_match_result_lead_line_first():
    out = axi_match_result(
        ["a.py:1: x", "a.py:2: y", "b.py:1: z"],
        total_matches=3,
        file_count=2,
        truncated=False,
        full_hint="include=",
    )
    first = out.splitlines()[0]
    assert first == "3 matches in 2 files"
    assert out.splitlines()[1] == "a.py:1: x"


def test_match_result_singular():
    out = axi_match_result(
        ["a.py:1: x"], total_matches=1, file_count=1, truncated=False, full_hint="h"
    )
    assert out.splitlines()[0] == "1 match in 1 file"


def test_match_result_truncated_signals():
    out = axi_match_result(
        ["l"] * 200, total_matches=200, file_count=5, truncated=True, full_hint="path="
    )
    assert "truncated" in out.splitlines()[0]
    assert "path=" in out.splitlines()[0]


def test_match_result_empty():
    assert axi_match_result([], total_matches=0, file_count=0, truncated=False, full_hint="h") == (
        "No matches. Next: h"
    )


def test_list_result_shows_truncation():
    out = axi_list_result(
        ["f1", "f2"], total=10, shown=2, noun="files", full_hint="path=", empty_hint="broaden"
    )
    assert out.splitlines()[0] == "10 files (showing 2 — narrow with path=)"


def test_list_result_no_truncation():
    out = axi_list_result(
        ["f1", "f2"], total=2, shown=2, noun="files", full_hint="path=", empty_hint="broaden"
    )
    assert out.splitlines()[0] == "2 files"


def test_file_header():
    assert axi_file_header("x.py", start=1, end=50, total_lines=200) == "x.py: lines 1-50 of 200"


def test_dir_header():
    assert axi_dir_header(dirs=2, files=3) == "5 entries (2 dirs, 3 files)"
    assert axi_dir_header(dirs=0, files=1) == "1 entry (0 dirs, 1 files)"
