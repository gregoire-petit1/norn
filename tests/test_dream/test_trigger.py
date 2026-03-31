"""Tests for the dream trigger (3-gate system)."""

from datetime import UTC, datetime, timedelta

import pytest

from norn.dream.trigger import DreamTrigger


@pytest.fixture
def memory_dir(tmp_path):
    d = tmp_path / "memory"
    d.mkdir()
    return d


@pytest.fixture
def trigger(memory_dir):
    return DreamTrigger(
        memory_dir=memory_dir,
        interval_hours=24,
        min_sessions=5,
    )


class TestTimeGate:
    def test_passes_when_no_previous_dream(self, trigger):
        """First dream ever: time gate passes."""
        assert trigger.check_time_gate() is True

    def test_passes_when_enough_time(self, trigger, memory_dir):
        """Enough time has passed since last dream."""
        last_dream = datetime.now(tz=UTC) - timedelta(hours=25)
        trigger.write_last_dream_time(last_dream)
        assert trigger.check_time_gate() is True

    def test_fails_when_too_recent(self, trigger, memory_dir):
        """Not enough time since last dream."""
        last_dream = datetime.now(tz=UTC) - timedelta(hours=1)
        trigger.write_last_dream_time(last_dream)
        assert trigger.check_time_gate() is False


class TestSessionGate:
    def test_passes_with_enough_sessions(self, trigger):
        """Enough sessions have accumulated."""
        assert trigger.check_session_gate(session_count=5) is True
        assert trigger.check_session_gate(session_count=10) is True

    def test_fails_with_too_few(self, trigger):
        """Not enough sessions."""
        assert trigger.check_session_gate(session_count=0) is False
        assert trigger.check_session_gate(session_count=4) is False


class TestLockGate:
    def test_passes_when_no_lock(self, trigger):
        """No other dream is running."""
        acquired = trigger.try_acquire_lock()
        assert acquired is True
        trigger.release_lock()

    def test_fails_when_locked(self, trigger):
        """Another dream holds the lock."""
        acquired1 = trigger.try_acquire_lock()
        assert acquired1 is True

        # Second attempt should fail (non-blocking)
        trigger2 = DreamTrigger(
            memory_dir=trigger._memory_dir,
            interval_hours=24,
            min_sessions=5,
        )
        acquired2 = trigger2.try_acquire_lock()
        assert acquired2 is False

        trigger.release_lock()

    def test_release_allows_reacquire(self, trigger):
        """After release, lock can be acquired again."""
        trigger.try_acquire_lock()
        trigger.release_lock()
        assert trigger.try_acquire_lock() is True
        trigger.release_lock()


class TestShouldDream:
    def test_all_gates_pass(self, trigger):
        """When all gates pass, should_dream returns True."""
        result = trigger.should_dream(session_count=5)
        assert result is True
        trigger.release_lock()

    def test_time_gate_blocks(self, trigger):
        """Recent dream blocks dreaming."""
        last_dream = datetime.now(tz=UTC) - timedelta(hours=1)
        trigger.write_last_dream_time(last_dream)
        result = trigger.should_dream(session_count=10)
        assert result is False

    def test_session_gate_blocks(self, trigger):
        """Too few sessions blocks dreaming."""
        result = trigger.should_dream(session_count=2)
        assert result is False

    def test_lock_gate_blocks(self, trigger):
        """Held lock blocks dreaming."""
        trigger.try_acquire_lock()
        trigger2 = DreamTrigger(
            memory_dir=trigger._memory_dir,
            interval_hours=24,
            min_sessions=5,
        )
        result = trigger2.should_dream(session_count=10)
        assert result is False
        trigger.release_lock()
