"""SMILES preprocessing utilities for PURS core."""

from __future__ import annotations


def normalize_smiles_text(smiles: str) -> str:
    """Normalize slash markers that historically caused parsing issues."""
    return smiles.replace("/", "").replace("\\", "")
