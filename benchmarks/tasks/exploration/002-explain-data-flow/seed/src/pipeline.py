from src.transforms import clean, enrich, validate


def ingest(raw_records: list[dict]) -> list[dict]:
    """Entry point: accept raw records."""
    return [r for r in raw_records if r.get("id") is not None]


def process(records: list[dict]) -> list[dict]:
    """Main processing pipeline."""
    cleaned = [clean(r) for r in records]
    validated = [r for r in cleaned if validate(r)]
    enriched = [enrich(r) for r in validated]
    return enriched


def export(records: list[dict]) -> str:
    """Format records as CSV string."""
    if not records:
        return ""
    headers = sorted(records[0].keys())
    lines = [",".join(headers)]
    for r in records:
        lines.append(",".join(str(r.get(h, "")) for h in headers))
    return "\n".join(lines)


def run_pipeline(raw: list[dict]) -> str:
    """Full pipeline: ingest -> process -> export."""
    ingested = ingest(raw)
    processed = process(ingested)
    return export(processed)
