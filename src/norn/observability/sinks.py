"""Output sinks — console (rich) and file (JSONL). Full impl in Task 3."""

from __future__ import annotations

import logging


def build_console_handler(shared_processors: list) -> logging.Handler:
    """Stub — full impl in Task 3."""
    return logging.NullHandler()


def build_file_handler(file_dir: str, shared_processors: list) -> logging.Handler:
    """Stub — full impl in Task 3."""
    return logging.NullHandler()
