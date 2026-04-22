def validate_age(age_str: str) -> int:
    """Validate and parse age string."""
    try:
        age = int(age_str)
    except ValueError:
        raise ValueError(f"Invalid age: {age_str}")
    if age < 0 or age > 150:
        raise ValueError(f"Age out of range: {age}")
    return age
