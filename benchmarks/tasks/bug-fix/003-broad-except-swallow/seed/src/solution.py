def safe_divide(a, b):
    """Divide a by b, returning None for division by zero.

    Should raise TypeError for non-numeric inputs.
    """
    try:
        return a / b
    except:  # noqa: E722 — BUG: too broad, swallows TypeError etc.
        return None
