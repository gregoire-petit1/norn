import pytest
from src.solution import fib


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (2, 1),
        (3, 2),
        (4, 3),
        (5, 5),
        (10, 55),
        (20, 6765),
    ],
)
def test_fib(n, expected):
    assert fib(n) == expected


def test_fib_negative():
    with pytest.raises((ValueError, NotImplementedError)):
        fib(-1)
