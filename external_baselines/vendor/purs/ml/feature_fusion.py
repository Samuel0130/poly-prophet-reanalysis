from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_RAW_COLUMNS = [
    "No.",
    "SMILES",
    "InchI",
    "ue/cm2·V-1·s-1",
    "uh/cm2·V-1·s-1",
    "PCE/%",
    "epsilon/F·m-1",
    "-HOMO/eV",
    "-LUMO/eV",
    "Eg/eV",
    "alpha",
    "mu",
]

DEFAULT_RAW_FEATURE_COLUMNS = [
    "-HOMO/eV",
    "-LUMO/eV",
    "Eg/eV",
]

DEFAULT_STANDARD_SCALAR_COLUMNS = [
    "homo",
    "lumo",
    "eg",
    "alpha",
    "mu",
]

MAX_SAFE_FEATURE_MAGNITUDE = 1e12


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"(?<=\d)\.$", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def stabilize_extreme_numeric_series(series: pd.Series, max_safe_magnitude: float = MAX_SAFE_FEATURE_MAGNITUDE) -> pd.Series:
    finite_values = series[np.isfinite(series)]
    if finite_values.empty:
        return series

    if finite_values.abs().max() <= max_safe_magnitude:
        return series

    # Some RDKit descriptors (for example Ipc) can reach astronomically large
    # values and overflow downstream float32 validation inside scikit-learn.
    return np.sign(series) * np.log10(np.abs(series) + 1.0)


def load_opecm_raw_table(raw_csv: str | Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_csv).copy()
    if len(raw_df.columns) != len(CANONICAL_RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(CANONICAL_RAW_COLUMNS)} OPECM raw columns, found {len(raw_df.columns)}."
        )
    raw_df.columns = CANONICAL_RAW_COLUMNS
    raw_df["sample_id"] = raw_df["No."].map(lambda value: str(int(value)))
    return raw_df


def build_opecm_raw_feature_table(
    raw_csv: str | Path,
    output_csv: str | Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    raw_df = load_opecm_raw_table(raw_csv)
    selected_columns = feature_columns or DEFAULT_RAW_FEATURE_COLUMNS
    feature_df = raw_df[["sample_id", *selected_columns]].copy()
    for column in selected_columns:
        feature_df[column] = clean_numeric_series(feature_df[column])
    feature_df = feature_df.dropna().drop_duplicates(subset=["sample_id"]).set_index("sample_id")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path)

    return feature_df


def build_opecm_descriptor_feature_table(
    descriptor_csv: str | Path,
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    descriptor_df = pd.read_csv(descriptor_csv).copy()
    descriptor_df = descriptor_df[descriptor_df["No"].notna()].copy()
    descriptor_df["sample_id"] = descriptor_df["No"].map(lambda value: str(int(value)))

    reserved_columns = {"No", "SMILES", "sample_id"}
    feature_columns = [column for column in descriptor_df.columns if column not in reserved_columns]
    feature_df = descriptor_df[["sample_id", *feature_columns]].copy()
    for column in feature_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df[column] = stabilize_extreme_numeric_series(feature_df[column])
    feature_df = feature_df.dropna().drop_duplicates(subset=["sample_id"]).set_index("sample_id")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path)

    return feature_df


def build_standard_scalar_feature_table(
    input_csv: str | Path,
    output_csv: str | Path | None = None,
    feature_columns: list[str] | None = None,
) -> pd.DataFrame:
    source_df = pd.read_csv(input_csv).copy()
    selected_columns = feature_columns or DEFAULT_STANDARD_SCALAR_COLUMNS
    required_columns = ["sample_id", *selected_columns]
    missing_columns = [column for column in required_columns if column not in source_df.columns]
    if missing_columns:
        raise KeyError(f"Missing columns in standard sample table: {missing_columns}")

    feature_df = source_df[required_columns].copy()
    feature_df["sample_id"] = feature_df["sample_id"].map(str)
    for column in selected_columns:
        feature_df[column] = pd.to_numeric(feature_df[column], errors="coerce")
        feature_df[column] = stabilize_extreme_numeric_series(feature_df[column])
    feature_df = feature_df.dropna().drop_duplicates(subset=["sample_id"]).set_index("sample_id")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.to_csv(output_path)

    return feature_df


def combine_feature_tables(
    base_feature_csv: str | Path,
    extra_feature_tables: list[pd.DataFrame] | list[str | Path],
    output_csv: str | Path | None = None,
) -> pd.DataFrame:
    base_df = pd.read_csv(base_feature_csv, index_col=0)
    base_df.index = base_df.index.map(str)

    combined = base_df.copy()
    for table in extra_feature_tables:
        if isinstance(table, (str, Path)):
            extra_df = pd.read_csv(table, index_col=0)
        else:
            extra_df = table.copy()
        extra_df.index = extra_df.index.map(str)
        combined = combined.join(extra_df, how="inner")

    if combined.empty:
        raise ValueError("No shared sample ids remained after feature-table fusion.")

    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path)

    return combined
