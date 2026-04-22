from src.solution import paginate


def test_first_page():
    items = list(range(1, 11))
    assert paginate(items, 1, 3) == [1, 2, 3]


def test_second_page():
    items = list(range(1, 11))
    assert paginate(items, 2, 3) == [4, 5, 6]


def test_last_page_partial():
    items = list(range(1, 11))
    assert paginate(items, 4, 3) == [10]


def test_page_beyond_range():
    items = list(range(1, 4))
    assert paginate(items, 2, 5) == []


def test_single_item_pages():
    items = ["a", "b", "c"]
    assert paginate(items, 1, 1) == ["a"]
    assert paginate(items, 2, 1) == ["b"]
    assert paginate(items, 3, 1) == ["c"]
