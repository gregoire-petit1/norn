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
    """Factory: bind redact keys at init_logging time."""

    def redact(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
        return redact_secrets(logger, method_name, event_dict, redact_keys=redact_keys)

    return redact


def redact_secrets(
    logger: Any,
    method_name: str,
    event_dict: dict[str, Any],
    redact_keys: list[str] | None = None,
) -> dict[str, Any]:
    """Replace sensitive field values with [REDACTED] (case-insensitive match)."""
    keys = redact_keys or []
    lowered = [k.lower() for k in keys]
    for key in list(event_dict.keys()):
        if any(pattern in key.lower() for pattern in lowered):
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
