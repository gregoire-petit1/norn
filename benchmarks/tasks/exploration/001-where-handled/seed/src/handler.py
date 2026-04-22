from src.parser import parse_int


def safe_parse(value: str) -> int | None:
    """Safely parse, catching ValueError."""
    try:
        return parse_int(value)
    except ValueError:
        return None


def batch_parse(values: list[str]) -> list[int]:
    """Parse a list, skipping invalid values."""
    results = []
    for v in values:
        try:
            results.append(parse_int(v))
        except ValueError:
            continue
    return results
