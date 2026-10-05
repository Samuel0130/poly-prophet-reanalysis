"""Graph dataset preparation helpers for polymer-unit graph workflows."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from purs.core.recognize import detect_name_column, detect_smiles_column


GRAPH_TARGET_CANDIDATES = [
    "PCE_max",
    "PCE_max(%)",
    "PCE/%",
    "graph_target",
    "target",
    "regression_target",
]


def detect_target_column(columns: list[str]) -> str | None:
    for candidate in GRAPH_TARGET_CANDIDATES:
        if candidate in columns:
            return candidate
    return None


def build_graph_input_csv(
    input_csv: str,
    output_dir: str | Path,
    name_column: str | None = None,
    smiles_column: str | None = None,
    target_column: str | None = None,
    exclude_ids: list[str] | None = None,
) -> dict:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)
    columns = df.columns.tolist()

    resolved_name_column = name_column or detect_name_column(columns)
    resolved_smiles_column = smiles_column or detect_smiles_column(columns)
    resolved_target_column = target_column or detect_target_column(columns)

    keep_columns = [resolved_name_column, resolved_smiles_column]
    rename_map = {
        resolved_name_column: "Compound ID",
        resolved_smiles_column: "smiles",
    }
    if resolved_target_column is not None:
        keep_columns.append(resolved_target_column)
        rename_map[resolved_target_column] = "PCE_max"

    graph_df = df[keep_columns].copy().rename(columns=rename_map)
    graph_df["Compound ID"] = graph_df["Compound ID"].astype(str)
    graph_df["smiles"] = graph_df["smiles"].astype(str)
    graph_df = graph_df.dropna(subset=["Compound ID", "smiles"])
    exclude_ids = [str(item) for item in (exclude_ids or [])]
    if exclude_ids:
        graph_df = graph_df[~graph_df["Compound ID"].isin(exclude_ids)].reset_index(drop=True)

    graph_input_csv = output_dir / "graph_input.csv"
    graph_df.to_csv(graph_input_csv, index=False)

    manifest = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "graph_input_csv": str(graph_input_csv),
        "row_count": int(len(graph_df)),
        "name_column": resolved_name_column,
        "smiles_column": resolved_smiles_column,
        "target_column": resolved_target_column,
        "normalized_columns": graph_df.columns.tolist(),
        "excluded_ids": exclude_ids,
    }
    manifest_path = output_dir / "graph_dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "graph_input_csv": graph_input_csv,
        "manifest_json": manifest_path,
        "row_count": len(graph_df),
        "name_column": resolved_name_column,
        "smiles_column": resolved_smiles_column,
        "target_column": resolved_target_column,
    }
