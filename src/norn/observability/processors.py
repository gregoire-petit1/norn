"""Custom structlog processors."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from norn.observability.events import EventName
from norn.observability.logger import session_id_var


def add_session_id(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject current session_id (if any) into the event."""
    sid = session_id_var.get()
    if sid is not None:
        event_dict["session_id"] = sid
    return event_dict


def add_timestamp_iso(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """Inject ISO-8601 UTC timestamp ending in 'Z'."""
    now = datetime.now(UTC).isoformat(timespec="microseconds")
    event_dict["timestamp"] = now.replace("+00:00", "Z")
    return event_dict


def make_redact_processor(redact_keys: list[str]):
    """Factory: bind redact keys at init_logging time.

    Takes a defensive copy so later mutation of the passed list by the caller
    does not affect the processor's behavior.
    """
    frozen_keys = list(redact_keys)

    def redact(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return redact_secrets(logger, method_name, event_dict, redact_keys=frozen_keys)

    return redact


def redact_secrets(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
    redact_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Replace values of sensitive fields with [REDACTED].

    Matching is exact on key name (case-insensitive). Users wanting
    fuzzy matching should add each variant explicitly to redact_keys,
    e.g. ['api_key', 'openai_api_key', 'x_api_key'].

    Note: only walks top-level keys of event_dict. Nested dicts/lists
    are not recursed. TODO(phase9-v2): add recursive redaction if events
    become nested. Tracked in
    docs/plans/2026-04-21-norn-phase9-observability-followups.md (item A3).

    Mutates event_dict in place and returns it (structlog processor convention).
    """
    keys = redact_keys or []
    lowered = {k.lower() for k in keys}
    for key in list(event_dict.keys()):
        if key.lower() in lowered:
            event_dict[key] = "[REDACTED]"
    return event_dict


def make_cost_processor(enabled: bool):
    """Factory: emit cost_usd on llm.complete events when enabled."""

    def add_cost(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        if not enabled:
            return event_dict
        if event_dict.get("event") != EventName.LLM_COMPLETE.value:
            return event_dict
        try:
            from litellm import completion_cost

            cost = completion_cost(
                model=event_dict.get("model", ""),
                prompt_tokens=event_dict.get("prompt_tokens", 0),
                completion_tokens=event_dict.get("completion_tokens", 0),
            )
            event_dict["cost_usd"] = round(float(cost), 6)
        except Exception:
            event_dict["cost_usd"] = None
        return event_dict

    return add_cost
