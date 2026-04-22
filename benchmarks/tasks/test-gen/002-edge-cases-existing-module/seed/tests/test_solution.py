from src.solution import RangeValidator


def test_basic_validate():
    rv = RangeValidator(0, 10)
    assert rv.validate(5) is True
    assert rv.validate(15) is False


def test_clamp():
    rv = RangeValidator(0, 10)
    assert rv.clamp(5) == 5
    assert rv.clamp(15) == 10
    assert rv.clamp(-5) == 0


# TODO: Add edge-case tests for boundary values, normalize, errors, etc.
