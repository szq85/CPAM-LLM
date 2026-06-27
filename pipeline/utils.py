from __future__ import annotations


def safe_str(x) -> str:
    """Convert a constraint entry (str, dict, or other) to a display string."""
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return x.get("name", x.get("id", x.get("type", str(x))))
    return str(x)
