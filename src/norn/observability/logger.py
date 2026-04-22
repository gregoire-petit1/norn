"""structlog configuration and session management."""

from __future__ import annotations

import contextlib
import logging
import os
import uuid
from contextvars import ContextVar
from pathlib import Path
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from norn.core.config import LoggingConfig

session_id_var: ContextVar[str | None] = ContextVar("norn_session_id", default=None)

# State file used to persist the most recent session UUID so the CLI
# (e.g. ``norn logs tail --session last``) can resolve aliases without the
# user having to copy-paste a UUID.
_STATE_DIR = Path.home() / ".norn" / "state"
_LAST_SESSION_FILE = _STATE_DIR / "last_session"


def new_session() -> str:
    """Generate a new session UUID, bind to contextvar, persist to state file."""
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    _persist_last_session(sid)
    return sid


def _persist_last_session(sid: str) -> None:
    """Best-effort write of the latest session id; never raises.

    Uses an atomic ``tmp`` + ``replace`` write so concurrent ``norn``
    processes can't corrupt the file (last writer wins, file is always
    either fully old or fully new). Any ``OSError`` (read-only fs, disk
    full, permission denied) is silently swallowed: the state file is a
    convenience for the CLI, not a correctness requirement of logging.
    """
    try:
        _STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _LAST_SESSION_FILE.with_suffix(".tmp")
        tmp.write_text(sid, encoding="utf-8")
        tmp.replace(_LAST_SESSION_FILE)
    except OSError:
        # fail-open: never let observability state-keeping crash the agent
        pass


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger for the given module."""
    return structlog.get_logger(name)


def init_logging(config: LoggingConfig, cli_level_override: str | None = None) -> None:
    """Configure structlog + stdlib logging for Norn.

    Idempotent: safe to call multiple times; subsequent calls replace handlers.
    If config.enabled is False, logging is silenced (no handlers attached).

    Console noise control: when no ``cli_level_override`` is given (normal
    mode), the console handler is raised to WARNING so the user only sees
    errors and the file sink captures everything at INFO. ``--verbose``
    forces the console back to DEBUG for developer use.
    """
    # Avoid duplicate handler registration on re-init
    root = logging.getLogger()
    for h in list(root.handlers):
        with contextlib.suppress(Exception):
            h.close()  # fail-open: never let cleanup crash logging init
        root.removeHandler(h)

    if not config.enabled:
        root.setLevel(logging.CRITICAL + 1)  # silence everything
        # structlog's make_filtering_bound_logger only accepts canonical levels
        # (10/20/30/40/50). CRITICAL (50) combined with root.setLevel(CRITICAL+1)
        # ensures every log call is silently dropped.
        structlog.configure(
            processors=[structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
            cache_logger_on_first_use=False,  # allow re-config in tests
        )
        return

    # Resolve level: CLI override > env > config
    level_str = (cli_level_override or os.environ.get("NORN_LOG_LEVEL") or config.level).upper()
    level = getattr(logging, level_str, logging.INFO)
    root.setLevel(level)

    # Lazy imports to avoid circular imports at module load
    from norn.observability.processors import (
        add_session_id,
        add_timestamp_iso,
        make_cost_processor,
        make_redact_processor,
    )
    from norn.observability.sinks import (
        build_console_handler,
        build_file_handler,
    )

    # Shared processor chain (runs before final rendering).
    # ``merge_contextvars`` MUST run first so fields bound via
    # ``structlog.contextvars.bind_contextvars`` / ``bound_contextvars``
    # (e.g. the router's ``tier``) are present in the event dict before
    # any downstream processor (cost, redact, …) inspects it.
    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        add_timestamp_iso,
        add_session_id,
        make_cost_processor(config.include_cost),
        make_redact_processor(config.redact_keys),
    ]

    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=False,  # allow re-config in tests
    )

    # Attach sinks
    if config.output in {"console", "both"}:
        # In normal mode (no --verbose), suppress INFO/DEBUG on the console
        # so the user only sees the response + metrics. File sink still
        # captures everything at the resolved level.
        console_level = None if cli_level_override else logging.WARNING
        root.addHandler(build_console_handler(shared_processors, level=console_level))
    if config.output in {"file", "both"}:
        root.addHandler(build_file_handler(config.file_dir, shared_processors))
