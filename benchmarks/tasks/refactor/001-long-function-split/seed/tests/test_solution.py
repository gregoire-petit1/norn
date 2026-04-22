import pytest
from src.solution import process_order


def _base_order(**overrides):
    order = {
        "customer_name": "Alice",
        "items": [{"name": "Widget", "price": 10.0, "quantity": 2}],
        "shipping_address": "123 Main St",
    }
    order.update(overrides)
    return order


def test_basic_order():
    result = process_order(_base_order())
    assert result["customer"] == "Alice"
    assert result["subtotal"] == 20.0
    assert result["item_count"] == 2


def test_missing_field():
    with pytest.raises(ValueError, match="Missing required"):
        process_order({"customer_name": "Bob"})


def test_empty_items():
    with pytest.raises(ValueError, match="at least one item"):
        process_order(_base_order(items=[]))


def test_coupon_save10():
    result = process_order(_base_order(coupon="SAVE10"))
    assert result["discount"] == 2.0


def test_coupon_save20():
    result = process_order(
        _base_order(
            items=[{"name": "Gadget", "price": 50.0, "quantity": 1}],
            coupon="SAVE20",
        )
    )
    assert result["discount"] == 10.0


def test_auto_discount_over_100():
    result = process_order(
        _base_order(
            items=[{"name": "Expensive", "price": 60.0, "quantity": 2}],
        )
    )
    assert result["discount"] == 6.0  # 5% of 120


def test_free_shipping_over_10_items():
    result = process_order(
        _base_order(
            items=[{"name": "Bulk", "price": 1.0, "quantity": 11}],
        )
    )
    assert result["shipping"] == 0.0


def test_tax_calculation():
    result = process_order(_base_order())
    # subtotal=20, discount=0, taxable=20, tax=1.6
    assert result["tax"] == 1.6


def test_negative_price():
    with pytest.raises(ValueError, match="Negative price"):
        process_order(
            _base_order(
                items=[{"name": "Bad", "price": -5.0, "quantity": 1}],
            )
        )
