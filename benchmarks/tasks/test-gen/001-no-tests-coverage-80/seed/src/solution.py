class Stack:
    """A simple stack implementation."""

    def __init__(self, max_size: int = 100) -> None:
        self._items: list = []
        self._max_size = max_size

    def push(self, item) -> None:
        if len(self._items) >= self._max_size:
            raise OverflowError("Stack is full")
        self._items.append(item)

    def pop(self):
        if self.is_empty():
            raise IndexError("Pop from empty stack")
        return self._items.pop()

    def peek(self):
        if self.is_empty():
            raise IndexError("Peek at empty stack")
        return self._items[-1]

    def is_empty(self) -> bool:
        return len(self._items) == 0

    def size(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        self._items.clear()

    def __contains__(self, item) -> bool:
        return item in self._items

    def __repr__(self) -> str:
        return f"Stack({self._items})"
