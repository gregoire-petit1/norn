def process_order(order: dict) -> dict:
    """Process an order: validate, calculate totals, apply discounts, format.

    This function is intentionally too long and should be refactored.
    """
    # Validate required fields
    required = ["customer_name", "items", "shipping_address"]
    for field in required:
        if field not in order:
            raise ValueError(f"Missing required field: {field}")
    if not order["items"]:
        raise ValueError("Order must have at least one item")
    for item in order["items"]:
        if "name" not in item or "price" not in item or "quantity" not in item:
            raise ValueError(f"Invalid item: {item}")
        if item["price"] < 0:
            raise ValueError(f"Negative price for {item['name']}")
        if item["quantity"] < 1:
            raise ValueError(f"Invalid quantity for {item['name']}")

    # Calculate subtotal
    subtotal = 0.0
    for item in order["items"]:
        subtotal += item["price"] * item["quantity"]

    # Apply discount
    discount = 0.0
    if order.get("coupon") == "SAVE10":
        discount = subtotal * 0.10
    elif order.get("coupon") == "SAVE20":
        discount = subtotal * 0.20
    elif subtotal > 100:
        discount = subtotal * 0.05

    # Calculate tax
    tax_rate = 0.08
    taxable = subtotal - discount
    tax = taxable * tax_rate

    # Calculate shipping
    total_items = sum(item["quantity"] for item in order["items"])
    if total_items > 10:
        shipping = 0.0  # free shipping
    elif subtotal > 50:
        shipping = 5.0
    else:
        shipping = 10.0

    # Build result
    total = taxable + tax + shipping
    return {
        "customer": order["customer_name"],
        "subtotal": round(subtotal, 2),
        "discount": round(discount, 2),
        "tax": round(tax, 2),
        "shipping": round(shipping, 2),
        "total": round(total, 2),
        "item_count": total_items,
    }
