"""Tests for logger setup and session_id contextvar."""

from __future__ import annotations

import asyncio

import pytest

from norn.core.config import LoggingConfig
from norn.observability.logger import (
    get_logger,
    init_logging,
    new_session,
    session_id_var,
)


@pytest.fixture(autouse=True)
def reset_session():
    """Reset session_id_var between tests."""
    token = session_id_var.set(None)
    yield
    session_id_var.reset(token)


def test_new_session_generates_uuid():
    sid = new_session()
    assert isinstance(sid, str)
    assert len(sid) == 36  # UUID4 format


def test_new_session_sets_contextvar():
    sid = new_session()
    assert session_id_var.get() == sid


def test_session_id_propagates_across_await():
    sid_holder = {}

    async def inner():
        sid_holder["inner"] = session_id_var.get()

    async def outer():
        sid = new_session()
        sid_holder["outer"] = sid
        await inner()

    asyncio.run(outer())
    assert sid_holder["outer"] == sid_holder["inner"]


def test_init_logging_idempotent(tmp_path):
    """Calling init_logging multiple times must not duplicate handlers."""
    import logging

    cfg = LoggingConfig(enabled=True, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    handler_count = len(logging.getLogger().handlers)
    init_logging(cfg)
    init_logging(cfg)
    assert len(logging.getLogger().handlers) == handler_count


def test_session_id_isolated_across_task_group():
    """Each task in a TaskGroup sees its own session_id.

    Phase 7's CoordinatorEngine uses TaskGroup; this guards against
    leakage where one worker's session_id bleeds into another.
    """
    results: dict[str, str] = {}

    async def worker(name: str):
        new_session()  # each task gets its own UUID
        results[name] = session_id_var.get()

    async def main():
        async with asyncio.TaskGroup() as tg:
            tg.create_task(worker("a"))
            tg.create_task(worker("b"))

    asyncio.run(main())
    assert results["a"] != results["b"]
    assert len(results["a"]) == 36
    assert len(results["b"]) == 36


def test_get_logger_returns_bound_logger():
    log = get_logger("test")
    assert hasattr(log, "info")
    assert hasattr(log, "bind")


def test_init_logging_disabled_no_op(tmp_path):
    cfg = LoggingConfig(enabled=False, output="file", file_dir=str(tmp_path))
    init_logging(cfg)
    log = get_logger("test")
    log.info("test.event", foo="bar")
    # No file should be created (disabled = no writes)
    files = list(tmp_path.iterdir())
    assert files == []
