"""Tests for EventName enum."""

from norn.observability.events import EventName


def test_event_names_stable():
    assert EventName.AGENT_RUN == "agent.run"
    assert EventName.LLM_COMPLETE == "llm.complete"
    assert EventName.TOOL_CALL == "tool.call"
    assert EventName.PERMISSION_DECISION == "permission.decision"
    assert EventName.ROUTING_DECISION == "routing.decision"
    assert EventName.FALLBACK == "fallback"
    assert EventName.INVARIANT_VIOLATION == "invariant.violation"
    assert EventName.LLM_EXCHANGE == "llm.exchange"
    assert EventName.SANDBOX_DECISION == "sandbox.decision"


def test_event_name_is_str():
    assert isinstance(EventName.AGENT_RUN, str)
    assert EventName.AGENT_RUN.value == "agent.run"


def test_event_names_unique():
    """Guards against accidental duplicate string values on the stable contract."""
    values = [e.value for e in EventName]
    assert len(values) == len(set(values))
    assert len(values) == 9
