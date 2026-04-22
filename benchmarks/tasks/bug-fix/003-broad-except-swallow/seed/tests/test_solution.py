import pytest
from src.solution import safe_divide


def test_normal_division():
    assert safe_divide(10, 2) == 5.0


def test_division_by_zero():
    assert safe_divide(10, 0) is None


def test_float_division():
    assert safe_divide(7, 2) == 3.5


def test_type_error_propagates():
    with pytest.raises(TypeError):
        safe_divide("hello", 2)


def test_type_error_propagates_second_arg():
    with pytest.raises(TypeError):
        safe_divide(10, "world")
