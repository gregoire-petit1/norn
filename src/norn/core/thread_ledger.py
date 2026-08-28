"""Append-only conversation thread invariant (SOTA v2, workstream A — W1.2).

Inspired by DeepSeek Harness: the session thread is the authoritative
artifact, and what the model sees must always be derivable from it. Norn's
``AgentLoop.history`` is append-only *de facto* (only ``.append()`` call
sites); this module makes that a checked invariant.

Design — proportionate, not dsh's two-full-serializations-per-dispatch:

- On ``extend()``, each *new* message is hashed once (sha256 chained on the
  previous digest). A message is hashed exactly once in its lifetime.
- On ``verify()``, the production fast path compares object identities
  (``id()``) per position — O(n) integer compares, zero serialization.
  A replaced/removed/reordered message is always caught; an in-place
  mutation of a message's fields is caught only in strict mode.
- Strict mode (``NORN_STRICT_INVARIANTS=1``, enabled autouse in the test
  suite) re-hashes the full chain — the dsh-grade byte-level guarantee,
  confined to CI/tests where the cost is irrelevant.

Failure policy: callers treat violations as fail-open in production (log an
``invariant.violation`` event, keep going) and fail-hard in strict mode
(``ThreadInvariantError``).
"""

from __future__ import annotations

import hashlib
import os

from norn.core.models import Message, Role

_STRICT_ENV_VAR = "NORN_STRICT_INVARIANTS"


class ThreadInvariantError(RuntimeError):
    """The conversation thread was mutated, truncated, or reordered."""


def canonical_message_digest(msg: Message, prev: str = "") -> str:
    """Chained digest: sha256(prev_digest + canonical JSON of the message)."""
    payload = prev + msg.model_dump_json(exclude_none=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class ThreadLedger:
    """Tracks the append-only history of one conversation thread."""

    def __init__(self, *, strict: bool | None = None) -> None:
        if strict is None:
            strict = os.environ.get(_STRICT_ENV_VAR) == "1"
        self.strict = strict
        # One entry per message ever appended: (id(message), chained digest).
        self._entries: list[tuple[int, str]] = []

    def __len__(self) -> int:
        return len(self._entries)

    def extend(self, history: list[Message]) -> None:
        """Record messages appended since the last call (hash new ones only)."""
        known = len(self._entries)
        prev = self._entries[-1][1] if self._entries else ""
        for msg in history[known:]:
            prev = canonical_message_digest(msg, prev)
            self._entries.append((id(msg), prev))

    def verify(self, history: list[Message]) -> list[str]:
        """Check ``history`` against the ledger. Returns violation strings.

        Production: length + per-position identity compare; a swapped-in
        object at a position is re-hashed and compared against the chain.
        Strict: additionally re-hashes the entire chain (catches in-place
        field mutations) and raises :class:`ThreadInvariantError` on any
        violation.
        """
        violations: list[str] = []

        if len(history) < len(self._entries):
            violations.append(
                f"history truncated: {len(history)} messages, ledger has {len(self._entries)}"
            )
        else:
            if self.strict:
                # Full re-hash of the chain: byte-level guarantee.
                prev = ""
                for i, (_, digest) in enumerate(self._entries):
                    prev = canonical_message_digest(history[i], prev)
                    if prev != digest:
                        violations.append(f"message [{i}] mutated (digest mismatch)")
                        break
            else:
                # Fast path: identity per position; re-hash only on id change
                # (a legitimately equal replacement object still verifies).
                for i, (obj_id, digest) in enumerate(self._entries):
                    if id(history[i]) == obj_id:
                        continue
                    prev = self._entries[i - 1][1] if i > 0 else ""
                    if canonical_message_digest(history[i], prev) != digest:
                        violations.append(f"message [{i}] replaced (digest mismatch)")
                        break

        if violations and self.strict:
            raise ThreadInvariantError("; ".join(violations))
        return violations

    def verify_derivation(self, outbound: list[Message], history: list[Message]) -> list[str]:
        """Check that every outbound message is derived from the thread.

        Every non-SYSTEM outbound message must be identity-member of
        ``history`` — the windowed view may drop old turns and inject a
        SYSTEM summary, but must never invent or rewrite user/assistant/tool
        turns. O(n), zero serialization.
        """
        history_ids = {id(m) for m in history}
        violations = [
            f"outbound message [{i}] (role={m.role.value}) not derived from thread history"
            for i, m in enumerate(outbound)
            if m.role != Role.SYSTEM and id(m) not in history_ids
        ]
        if violations and self.strict:
            raise ThreadInvariantError("; ".join(violations))
        return violations
