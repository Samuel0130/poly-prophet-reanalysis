"""Input/output helpers for unified PURS workflows."""

from __future__ import annotations

from pathlib import Path


def ensure_output_dir(path: str | Path) -> Path:
    """Create and return an output directory path."""
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
