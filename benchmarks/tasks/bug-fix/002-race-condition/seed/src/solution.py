class Counter:
    """A counter that should be thread-safe but isn't."""

    def __init__(self) -> None:
        self._value = 0

    def increment(self) -> None:
        # BUG: not thread-safe — read-modify-write without lock
        current = self._value
        self._value = current + 1

    @property
    def value(self) -> int:
        return self._value
