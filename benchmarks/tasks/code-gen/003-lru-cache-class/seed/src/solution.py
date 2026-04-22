class LRUCache:
    """Least Recently Used cache with fixed capacity."""

    def __init__(self, capacity: int) -> None:
        raise NotImplementedError

    def get(self, key: str) -> int | None:
        raise NotImplementedError

    def put(self, key: str, value: int) -> None:
        raise NotImplementedError
