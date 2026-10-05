"""Shared adapter for legacy PU-MPNN preprocessing utilities."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd

from purs.graph.pu_mpnn.legacy.API import bondFeatures2 as legacy_bond_features2
from purs.graph.pu_mpnn.legacy.API import get_mpnn_input as legacy_get_mpnn_input


def _write_temp_csv(df: pd.DataFrame):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8", newline="") as tmp:
        df.to_csv(tmp.name, index=False)
        return tmp.name


def _write_skip_report(source_path, skipped_rows):
    if not skipped_rows:
        return None
    source_path = Path(source_path)
    report_path = source_path.with_name(f"{source_path.stem}_legacy_mpnn_skipped.csv")
    pd.DataFrame(skipped_rows).to_csv(report_path, index=False)
    return report_path


def get_mpnn_input(csv_name, sdf_name):
    """Centralized adapter to the legacy PU-MPNN preprocessing entry point."""

    try:
        return legacy_get_mpnn_input(csv_name, sdf_name)
    except Exception as bulk_exc:
        df = pd.read_csv(csv_name)
        compatible_rows = []
        skipped_rows = []

        for _, row in df.iterrows():
            row_df = pd.DataFrame([row])
            row_csv = _write_temp_csv(row_df)
            row_sdf_handle = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
            row_sdf_handle.close()
            row_sdf = row_sdf_handle.name
            try:
                legacy_get_mpnn_input(row_csv, row_sdf)
                compatible_rows.append(row.to_dict())
            except Exception as row_exc:
                skipped_rows.append(
                    {
                        "name": row.get("name"),
                        "smiles": row.get("smiles"),
                        "target": row.get("target"),
                        "error_type": type(row_exc).__name__,
                        "error_message": str(row_exc),
                    }
                )
            finally:
                if os.path.exists(row_csv):
                    os.unlink(row_csv)
                if os.path.exists(row_sdf):
                    os.unlink(row_sdf)

        if not compatible_rows:
            raise RuntimeError(
                f"Legacy PU-MPNN preprocessing failed for every row in {csv_name}. "
                f"First bulk error: {type(bulk_exc).__name__}: {bulk_exc}"
            ) from bulk_exc

        report_path = _write_skip_report(csv_name, skipped_rows)
        filtered_df = pd.DataFrame(compatible_rows, columns=df.columns)
        filtered_csv = _write_temp_csv(filtered_df)
        try:
            result = legacy_get_mpnn_input(filtered_csv, sdf_name)
        finally:
            if os.path.exists(filtered_csv):
                os.unlink(filtered_csv)

        message = (
            f"[legacy_mpnn_adapter] bulk preprocessing failed for {csv_name}; "
            f"recovered {len(compatible_rows)} rows with per-row fallback and skipped {len(skipped_rows)} rows. "
            f"First bulk error: {type(bulk_exc).__name__}: {bulk_exc}"
        )
        if report_path is not None:
            message += f" Skip report: {report_path}"
        print(message)
        return result


def bond_features2(pos, bid1, bid2, mol, rings):
    """Expose the legacy edge-feature helper through one stable adapter."""
    return legacy_bond_features2(pos, bid1, bid2, mol, rings)
