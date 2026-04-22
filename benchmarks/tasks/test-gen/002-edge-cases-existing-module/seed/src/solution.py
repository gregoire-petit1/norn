class RangeValidator:
    """Validates that values fall within a configured range."""

    def __init__(self, min_val: float, max_val: float) -> None:
        if min_val >= max_val:
            raise ValueError("min_val must be less than max_val")
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, value: float) -> bool:
        return self.min_val <= value <= self.max_val

    def clamp(self, value: float) -> float:
        return max(self.min_val, min(self.max_val, value))

    def normalize(self, value: float) -> float:
        if not self.validate(value):
            raise ValueError(f"{value} outside range [{self.min_val}, {self.max_val}]")
        span = self.max_val - self.min_val
        return (value - self.min_val) / span

    def __repr__(self) -> str:
        return f"RangeValidator({self.min_val}, {self.max_val})"
