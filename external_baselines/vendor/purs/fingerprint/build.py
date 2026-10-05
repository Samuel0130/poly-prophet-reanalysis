"""Fingerprint-building entry points."""

from __future__ import annotations

from purs.core.classify import classify_units
from purs.core.recognize import recognize_units


def build_pufp(input_csv, name_column=None, smiles_column=None, output_dir=".", strict=False):
    """Run the standard PURS recognition + classification chain for PUFp-ready outputs."""
    recognize_result = recognize_units(
        input_csv=input_csv,
        name_column=name_column,
        smiles_column=smiles_column,
        output_dir=output_dir,
        strict=strict,
    )
    classify_result = classify_units(
        ring_total_list_path=recognize_result["output_dir"] / "ring_total_list.csv",
        index_data_path=recognize_result["output_dir"] / "index_data.csv",
        output_dir=recognize_result["output_dir"],
    )
    return {
        "output_dir": recognize_result["output_dir"],
        "name_column": recognize_result["name_column"],
        "smiles_column": recognize_result["smiles_column"],
        "total_rows": recognize_result["total_rows"],
        "sample_count": recognize_result["sample_count"],
        "skipped_count": recognize_result["skipped_count"],
        "unit_count": recognize_result["unit_count"],
        "ring_count": classify_result["ring_count"],
        "skipped_records_csv": recognize_result["skipped_records_csv"],
        "run_summary_json": recognize_result["run_summary_json"],
    }
