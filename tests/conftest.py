"""Shared pytest fixtures for the Norn test suite.

This conftest centralises hygiene for observability side effects so individual
test modules don't need to repeat the pattern. Specifically:

- Redirect ``LoggingConfig.file_dir`` default to a per-test ``tmp_path`` so that
  any code path calling ``init_logging()`` with a default config does NOT write
  to the real ``~/.norn/logs/`` directory during the test suite.
- Reset the observability session contextvar between tests to prevent
  cross-test leakage of session IDs.

Individual tests that explicitly construct ``LoggingConfig(file_dir=...)`` are
unaffected (their explicit value wins over the patched default).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _redirect_default_log_dir(tmp_path: Path, request: pytest.FixtureRequest) -> Iterator[None]:
    """Redirect the default ``LoggingConfig.file_dir`` to ``tmp_path``.

    Autouse so every test in the suite gets filesystem isolation for logs
    without opt-in. Tests that pass an explicit ``file_dir`` keep their value
    because Pydantic only consults the field default when no value is supplied.

    Opt-out: mark a test with ``@pytest.mark.no_log_redirect`` when the test
    asserts on the real production default (e.g. ``test_logging_config_defaults``).

    Implementation detail: Pydantic v2 compiles ``__init__`` from field defaults
    at class-definition time, so mutating ``FieldInfo.default`` alone is not
    enough — we must call ``model_rebuild(force=True)`` to regenerate the
    validator. Both ``LoggingConfig`` and the parent ``NornConfig`` need a
    rebuild because ``NornConfig.logging`` has its own compiled default
    (``LoggingConfig()`` evaluated once at class-definition time).
    """
    if request.node.get_closest_marker("no_log_redirect") is not None:
        yield
        return

    from norn.core.config import LoggingConfig, NornConfig

    log_dir = tmp_path / "norn-logs"
    logging_field = LoggingConfig.model_fields["file_dir"]
    original_file_dir = logging_field.default
    logging_field.default = str(log_dir)
    LoggingConfig.model_rebuild(force=True)

    # NornConfig.logging default is a LoggingConfig instance captured at class
    # definition time — rebuild it too so ``NornConfig()`` picks up the new dir.
    nornconfig_logging_field = NornConfig.model_fields["logging"]
    original_logging_default = nornconfig_logging_field.default
    nornconfig_logging_field.default = LoggingConfig()
    NornConfig.model_rebuild(force=True)

    try:
        yield
    finally:
        logging_field.default = original_file_dir
        LoggingConfig.model_rebuild(force=True)
        nornconfig_logging_field.default = original_logging_default
        NornConfig.model_rebuild(force=True)


@pytest.fixture(autouse=True)
def _reset_session_contextvar() -> Iterator[None]:
    """Ensure the session_id contextvar is reset between tests.

    Prevents a session UUID set by one test from leaking into another when
    tests run in the same event loop / thread.
    """
    from norn.observability.logger import session_id_var

    token = session_id_var.set(None)
    try:
        yield
    finally:
        session_id_var.reset(token)
