def clean(record: dict) -> dict:
    """Strip whitespace from string fields and lowercase the name."""
    result = {}
    for k, v in record.items():
        if isinstance(v, str):
            result[k] = v.strip().lower() if k == "name" else v.strip()
        else:
            result[k] = v
    return result


def validate(record: dict) -> bool:
    """Check that record has required fields and valid values."""
    if "id" not in record or "name" not in record:
        return False
    if not record["name"]:
        return False
    return True


def enrich(record: dict) -> dict:
    """Add computed fields to the record."""
    record["name_length"] = len(record.get("name", ""))
    record["processed"] = True
    return record
