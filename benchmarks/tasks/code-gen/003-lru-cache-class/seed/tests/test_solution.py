from src.solution import LRUCache


def test_basic_put_get():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    assert cache.get("a") == 1
    assert cache.get("b") == 2


def test_eviction():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.put("c", 3)  # evicts "a"
    assert cache.get("a") is None
    assert cache.get("b") == 2
    assert cache.get("c") == 3


def test_access_refreshes():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("b", 2)
    cache.get("a")  # refresh "a"
    cache.put("c", 3)  # evicts "b" (not "a")
    assert cache.get("a") == 1
    assert cache.get("b") is None


def test_update_existing():
    cache = LRUCache(2)
    cache.put("a", 1)
    cache.put("a", 10)
    assert cache.get("a") == 10


def test_missing_key():
    cache = LRUCache(1)
    assert cache.get("x") is None
