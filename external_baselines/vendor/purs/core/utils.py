"""General helper functions shared across package layers."""

from __future__ import annotations


def deduplicate_name(name: str, duplicate_index: int) -> str:
    """Apply the package-wide duplicate naming rule."""
    if duplicate_index <= 0:
        return name
    return f"{name}-{duplicate_index}"
