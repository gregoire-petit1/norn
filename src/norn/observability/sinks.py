"""Output sinks — console (rich-rendered) and file (JSONL rotating daily)."""

from __future__ import annotations

import contextlib
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TextIO

import structlog


def build_console_handler(shared_processors: list) -> logging.Handler:
    """Colored rich-style console handler for dev ergonomics."""
    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )
    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(formatter)
    return handler


class DailyRotatingJsonlHandler(logging.Handler):
    """Writes JSON lines to <dir>/YYYY-MM-DD.jsonl (UTC), rotating at day boundary.

    Designed for low-volume observability events. For high-volume workloads,
    consider switching to an external log shipper. Opens the file lazily on
    first emit and re-opens when the UTC date changes.

    Thread-safety: relies on ``logging.Handler.handle()`` acquiring ``self.lock``
    around each ``emit()`` call. Do NOT call ``emit()`` directly from user code.

    Multi-process: uses append mode so concurrent processes will not truncate
    each other's data. However, POSIX only guarantees atomic appends up to
    ``PIPE_BUF`` (typically 4096 bytes). Records larger than that may be
    interleaved under concurrent writers. Norn is single-process today; revisit
    if that assumption changes.
    """

    def __init__(self, dir_path: Path):
        super().__init__()
        self._dir_path = dir_path
        self._current_date: str | None = None
        self._fh: TextIO | None = None

    def _ensure_file(self) -> None:
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        if today != self._current_date:
            self._close_fh()
            self._dir_path.mkdir(parents=True, exist_ok=True)
            path = self._dir_path / f"{today}.jsonl"
            # Append so multiple processes or re-inits don't clobber existing lines
            self._fh = path.open("a", encoding="utf-8")
            self._current_date = today

    def _close_fh(self) -> None:
        if self._fh is not None:
            with contextlib.suppress(Exception):
                self._fh.close()  # fail-open: never let log teardown crash the app
            self._fh = None

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._ensure_file()
            if self._fh is None:
                # _ensure_file should always set _fh; guard defensively rather than
                # assert (which disappears under `python -O`).
                raise RuntimeError("log file handle not initialized")
            msg = self.format(record)
            self._fh.write(msg + "\n")
            self._fh.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        self._close_fh()
        super().close()


def build_file_handler(file_dir: str, shared_processors: list) -> logging.Handler:
    """Daily-rotating JSONL file handler. Expands ~ in the path."""
    dir_path = Path(file_dir).expanduser()
    dir_path.mkdir(parents=True, exist_ok=True)

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    handler = DailyRotatingJsonlHandler(dir_path)
    handler.setFormatter(formatter)
    return handler
