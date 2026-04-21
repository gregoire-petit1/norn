"""structlog configuration and session management."""

from __future__ import annotations

import logging
import os
import uuid
from contextvars import ContextVar
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from norn.core.config import LoggingConfig

session_id_var: ContextVar[str | None] = ContextVar("norn_session_id", default=None)

_configured: bool = False


def new_session() -> str:
    """Generate a new session UUID and bind to contextvar."""
    sid = str(uuid.uuid4())
    session_id_var.set(sid)
    return sid


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a structlog BoundLogger for the given module."""
    return structlog.get_logger(name)


def init_logging(config: LoggingConfig, cli_level_override: str | None = None) -> None:
    """Configure structlog + stdlib logging for Norn.

    Idempotent: safe to call multiple times; subsequent calls replace handlers.
    If config.enabled is False, logging is silenced (no handlers attached).
    """
    global _configured

    # Avoid duplicate handler registration on re-init
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)

    if not config.enabled:
        root.setLevel(logging.CRITICAL + 1)  # silence everything
        structlog.configure(
            processors=[structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(logging.CRITICAL),
            cache_logger_on_first_use=False,  # allow re-config in tests
        )
        _configured = True
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

    # Shared processor chain (runs before final rendering)
    shared_processors: list = [
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
        root.addHandler(build_console_handler(shared_processors))
    if config.output in {"file", "both"}:
        root.addHandler(build_file_handler(config.file_dir, shared_processors))

    _configured = True
