import threading
from src.solution import Counter


def test_single_thread():
    c = Counter()
    for _ in range(100):
        c.increment()
    assert c.value == 100


def test_concurrent_increment():
    c = Counter()
    n_threads = 10
    increments_per_thread = 1000

    def worker():
        for _ in range(increments_per_thread):
            c.increment()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert c.value == n_threads * increments_per_thread


def test_initial_value():
    c = Counter()
    assert c.value == 0
