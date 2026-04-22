def paginate(items: list, page: int, per_page: int) -> list:
    """Return items for the given 1-based page number.

    Args:
        items: Full list of items.
        page: 1-based page number.
        per_page: Items per page.

    Returns:
        Slice of items for the requested page.
    """
    # BUG: off-by-one in start index
    start = page * per_page
    end = start + per_page
    return items[start:end]
