from src.solution import parse_csv


def test_simple_csv():
    text = "name,age\nAlice,30\nBob,25"
    result = parse_csv(text)
    assert len(result) == 2
    assert result[0] == {"name": "Alice", "age": "30"}
    assert result[1] == {"name": "Bob", "age": "25"}


def test_quoted_fields():
    text = 'name,city\n"Smith, John","New York"'
    result = parse_csv(text)
    assert result[0]["name"] == "Smith, John"
    assert result[0]["city"] == "New York"


def test_empty_csv():
    assert parse_csv("name,age") == []


def test_single_column():
    text = "item\napple\nbanana"
    result = parse_csv(text)
    assert len(result) == 2
    assert result[0] == {"item": "apple"}
