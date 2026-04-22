from src.solution import fizzbuzz


def test_fizzbuzz_15():
    result = list(fizzbuzz(15))
    assert len(result) == 15
    assert result[0] == "1"
    assert result[2] == "Fizz"
    assert result[4] == "Buzz"
    assert result[14] == "FizzBuzz"


def test_fizzbuzz_1():
    assert list(fizzbuzz(1)) == ["1"]


def test_fizzbuzz_types():
    for item in fizzbuzz(20):
        assert isinstance(item, str)
