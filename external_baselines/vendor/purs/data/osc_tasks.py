from __future__ import annotations

from pathlib import Path

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

PAPER54_CLASS3_LABELS = ["low", "medium", "high"]
PAPER67_CLASS4_LABELS = ["0_to_1", "1_to_4", "4_to_10", "gt_10"]


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.strip()
    cleaned = cleaned.str.replace(r"(?<=\d)\.$", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return pd.to_numeric(cleaned, errors="coerce")


def _safe_is_valid_smiles(smiles: str) -> bool:
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "RDKit is required to compute expected_valid in the OSC parent table."
        ) from exc

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        normalized = str(smiles).replace("/", "").replace("\\", "")
        return Chem.MolFromSmiles(normalized) is not None
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")


def dominant_carrier_label(ue: float | None, uh: float | None) -> str:
    if pd.isna(ue) or pd.isna(uh):
        return "unknown"
    if float(ue) > float(uh):
        return "n"
    if float(uh) > float(ue):
        return "p"
    return "tie"


def paper54_mobility_class3(mobility: float | None) -> str | None:
    if pd.isna(mobility):
        return None
    value = float(mobility)
    if value > 4.0:
        return "high"
    if value > 1.0:
        return "medium"
    if value >= 0.0:
        return "low"
    return None


def paper54_mobility_class3_id(mobility: float | None) -> int | None:
    label = paper54_mobility_class3(mobility)
    if label is None:
        return None
    return PAPER54_CLASS3_LABELS.index(label)


def paper67_mobility_class4(mobility: float | None) -> str | None:
    if pd.isna(mobility):
        return None
    value = float(mobility)
    if value > 10.0:
        return "gt_10"
    if value > 4.0:
        return "4_to_10"
    if value > 1.0:
        return "1_to_4"
    if value >= 0.0:
        return "0_to_1"
    return None


def paper67_mobility_class4_id(mobility: float | None) -> int | None:
    label = paper67_mobility_class4(mobility)
    if label is None:
        return None
    return PAPER67_CLASS4_LABELS.index(label)


def load_opecm_raw_table(raw_csv: str | Path) -> pd.DataFrame:
    raw_df = pd.read_csv(raw_csv).copy()
    if len(raw_df.columns) != len(CANONICAL_RAW_COLUMNS):
        raise ValueError(
            f"Expected {len(CANONICAL_RAW_COLUMNS)} OPECM raw columns, found {len(raw_df.columns)}."
        )
    raw_df.columns = CANONICAL_RAW_COLUMNS
    return raw_df


def build_opecm_osc_parent_table(raw_csv: str | Path) -> pd.DataFrame:
    raw_df = load_opecm_raw_table(raw_csv)

    for column in CANONICAL_RAW_COLUMNS[3:]:
        raw_df[column] = clean_numeric_series(raw_df[column])

    parent = pd.DataFrame(
        {
            "source_dataset": "OPECM",
            "source_record_id": raw_df["No."].astype(int),
            "sample_id": raw_df["No."].astype(int).astype(str),
            "smiles": raw_df["SMILES"].astype(str),
            "ue": raw_df["ue/cm2·V-1·s-1"],
            "uh": raw_df["uh/cm2·V-1·s-1"],
            "homo": raw_df["-HOMO/eV"],
            "lumo": raw_df["-LUMO/eV"],
            "eg": raw_df["Eg/eV"],
            "epsilon": raw_df["epsilon/F·m-1"],
            "alpha": raw_df["alpha"],
            "mu": raw_df["mu"],
        }
    )
    parent["expected_valid"] = parent["smiles"].map(_safe_is_valid_smiles)
    parent["notes"] = parent["expected_valid"].map(lambda flag: "valid" if flag else "invalid_smiles")

    parent["dominant_carrier"] = [
        dominant_carrier_label(ue, uh) for ue, uh in zip(parent["ue"], parent["uh"])
    ]
    parent["max_mobility"] = parent[["ue", "uh"]].max(axis=1, skipna=True)

    parent["paper54_carrier_type"] = parent["dominant_carrier"]
    parent["paper54_mobility_class3"] = parent["max_mobility"].map(paper54_mobility_class3)
    parent["paper54_mobility_class3_id"] = parent["max_mobility"].map(paper54_mobility_class3_id)

    parent["paper67_ue_class4"] = parent["ue"].map(paper67_mobility_class4)
    parent["paper67_ue_class4_id"] = parent["ue"].map(paper67_mobility_class4_id)
    parent["paper67_uh_class4"] = parent["uh"].map(paper67_mobility_class4)
    parent["paper67_uh_class4_id"] = parent["uh"].map(paper67_mobility_class4_id)

    parent["paper54_ready"] = (
        parent["expected_valid"]
        & parent["ue"].notna()
        & parent["uh"].notna()
        & parent["paper54_carrier_type"].isin(["n", "p", "tie"])
        & parent["paper54_mobility_class3"].notna()
    )
    parent["paper67_ready"] = (
        parent["expected_valid"]
        & parent["ue"].notna()
        & parent["uh"].notna()
        & parent["homo"].notna()
        & parent["lumo"].notna()
        & parent["paper67_ue_class4"].notna()
        & parent["paper67_uh_class4"].notna()
    )
    return parent


def build_paper_style_task_tables(parent_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    pufp_df = parent_df[parent_df["paper54_ready"]].copy()
    pugraph_df = parent_df[parent_df["paper67_ready"]].copy()
    return {
        "paper54_parent": parent_df,
        "paper54_tasks": pufp_df,
        "paper67_tasks": pugraph_df,
    }
