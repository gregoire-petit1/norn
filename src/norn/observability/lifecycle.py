"""Lifecycle instrumentation helper.

Provides :func:`measure_and_log`, an ``asynccontextmanager`` that wraps a
block of async work and emits a single structured event on exit with
``duration_ms``, ``success``, and — on failure — the harmonized
``error_type`` / ``error_message`` pair documented in
:mod:`norn.observability.events`.

Contract
--------
1. The caller receives a mutable ``dict`` and may populate it with
   dynamic fields (e.g. ``prompt_tokens``, ``finish_reason``).
2. On normal exit, the event is emitted at INFO with ``success=True``.
3. On exception, the event is emitted at ERROR with ``success=False`` and
   the exception is re-raised (control flow preserved).
4. Logging failures are swallowed — observability is fail-open and MUST
   NEVER affect caller control flow (disk full, permission denied,
   filesystem unmounted, …).

Static vs dynamic fields
------------------------
``static_fields`` are known at entry (e.g. ``model``, ``provider``);
dynamic fields are populated inside the block. Both are merged into the
final event; dynamic keys win on conflict.
"""

from __future__ import annotations

import contextlib
import time
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    import structlog


def _emit(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    level: str,
    payload: dict[str, Any],
) -> None:
    """Emit one event. Isolated to a module-level function so tests can
    patch it to simulate logging failures without reaching into structlog.
    """
    getattr(logger, level)(event, **payload)


@asynccontextmanager
async def measure_and_log(
    logger: structlog.stdlib.BoundLogger,
    event: str,
    *,
    duration_field: str = "duration_ms",
    **static_fields: Any,
) -> AsyncIterator[dict[str, Any]]:
    """Emit an instrumentation event with duration + success/error.

    Yields a mutable dict that the caller populates with dynamic fields
    (e.g. ``prompt_tokens``, ``finish_reason``). On exit, emits ``event``
    at INFO (success) or ERROR (failure) with the elapsed duration under
    ``duration_field`` (default ``"duration_ms"``; pass ``"latency_ms"``
    for LLM-style events), ``success``, the static fields, the dynamic
    fields, and — on failure — the harmonized ``error_type`` /
    ``error_message`` pair.

    Logging failures are suppressed — they never affect caller control
    flow. Exceptions from the wrapped block are re-raised unchanged.
    """
    dynamic: dict[str, Any] = {}
    start = time.monotonic()
    try:
        yield dynamic
    except Exception as exc:
        payload = {
            **static_fields,
            **dynamic,
            duration_field: int((time.monotonic() - start) * 1000),
            "success": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        # Fail-open by contract: logging failures must never propagate.
        with contextlib.suppress(Exception):
            _emit(logger, event, "error", payload)
        raise
    else:
        payload = {
            **static_fields,
            **dynamic,
            duration_field: int((time.monotonic() - start) * 1000),
            "success": True,
        }
        with contextlib.suppress(Exception):
            _emit(logger, event, "info", payload)
