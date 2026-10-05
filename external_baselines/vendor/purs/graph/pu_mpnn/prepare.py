"""Wrapper utilities for preparing legacy PU-MPNN inputs inside the unified package."""

from __future__ import annotations

import argparse
import csv
import pickle as pkl
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import sparse
from rdkit import Chem
from sklearn.metrics.pairwise import euclidean_distances

from purs.graph.legacy_mpnn_adapter import bond_features2, get_mpnn_input


def _resolve_column(columns, candidates, label):
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise KeyError(f"Could not find a supported {label} column. Tried: {candidates}")


def _make_rdkit_safe_path(path: Path) -> Path:
    try:
        str(path).encode("ascii")
        return path
    except UnicodeEncodeError:
        safe_root = Path(tempfile.gettempdir()) / "purs_rdkit_tmp"
        safe_root.mkdir(parents=True, exist_ok=True)
        return safe_root / path.name


def _normalize_sample_name(value) -> str:
    text = str(value).strip()
    if text.endswith(".0"):
        try:
            numeric = float(text)
            if numeric.is_integer():
                return str(int(numeric))
        except ValueError:
            pass
    return text


def build_release_check_csv(src_csv: Path, dst_csv: Path, limit: int | None = None):
    df = pd.read_csv(src_csv)
    if limit is not None and limit > 0:
        df = df.head(limit)
    df = df.copy()
    if "target" not in df.columns:
        target_column = next((col for col in ("PCE_max", "PCE_max(%)", "PCE/%") if col in df.columns), None)
        if target_column is None:
            df["target"] = np.arange(len(df), dtype=np.float64)
        else:
            df["target"] = df[target_column]
    id_column = _resolve_column(df.columns, ("name", "Compound ID", "sample_id", "No.", "No"), "id")
    smiles_column = _resolve_column(df.columns, ("smiles", "SMILES", "n(SMILES)", "p(SMILES)"), "smiles")
    out_df = df[[id_column, smiles_column, "target"]].rename(columns={id_column: "name", smiles_column: "smiles"})
    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst_csv, index=False)
    return out_df


def build_graph_pkl(csv_path: Path, sdf_path: Path, out_path: Path):
    csv_path = Path(csv_path).expanduser().resolve()
    sdf_path = Path(sdf_path).expanduser().resolve()
    out_path = Path(out_path).expanduser().resolve()
    sdf_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rdkit_sdf_path = _make_rdkit_safe_path(sdf_path)
    rdkit_sdf_path.parent.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(csv_path)
    feature_df["name"] = feature_df["name"].map(_normalize_sample_name)
    index_dict, pu_dict, pair_atom_dict, adj_list, pu_feature, structure, n_max, dim_node = get_mpnn_input(
        csv_path.as_posix(),
        rdkit_sdf_path.as_posix(),
    )
    pu_feature_df = pd.DataFrame(pu_feature)

    dim_edge = 8
    dv = []
    de = []
    dp = []
    dy = []
    dsmi = []
    skipped_empty_graphs = []

    for i, mol in enumerate(structure):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue

        name = _normalize_sample_name(mol.GetProp("_Name"))
        if name not in pu_dict:
            continue

        smi = Chem.MolToSmiles(mol)
        if "." in smi:
            continue

        num_index = list(pu_dict[name].values())
        num_node = len(num_index)
        node_name = list(pu_dict[name].keys())
        pair_atom = pair_atom_dict[name]
        adj = adj_list[i]
        if num_node == 0 or not node_name:
            skipped_empty_graphs.append(name)
            continue
        rings = mol.GetRingInfo().AtomRings()

        node = np.zeros((n_max, dim_node), dtype=np.int8)
        for j in range(num_node):
            index = num_index[j]
            node[j, :] = pu_feature_df.loc[index].to_numpy(dtype=np.int8)

        pos = mol.GetConformer().GetPositions()
        pos_array = np.array(pos)
        node_ave = []
        for j in range(num_node):
            index = node_name[j]
            atom_num = index_dict[name][index]
            ave_pos = pos_array[atom_num].mean(axis=0)
            node_ave.append(ave_pos)
        if not node_ave:
            skipped_empty_graphs.append(name)
            continue
        node_pos = np.array(node_ave)
        if node_pos.ndim != 2 or node_pos.shape[0] == 0:
            skipped_empty_graphs.append(name)
            continue
        proximity = np.zeros((n_max, n_max), dtype=np.float32)
        proximity[:num_node, :num_node] = euclidean_distances(node_pos)

        target_rows = feature_df.loc[feature_df["name"] == name, ["target"]]
        if target_rows.empty:
            continue
        target = target_rows.iloc[0].to_numpy(dtype=np.float64)

        edge = np.zeros((n_max, n_max, dim_edge), dtype=np.float32)
        pair_lookup = {}
        for pair_idx, pair_name in enumerate(pair_atom["pair_index"]):
            pair_lookup[tuple(pair_name)] = pair_atom["pair_atom"][pair_idx]

        for j in range(num_node):
            for k in range(num_node):
                if adj[j, k] != 1:
                    continue
                j_name = node_name[j]
                k_name = node_name[k]
                j_index = num_index[j]
                k_index = num_index[k]
                pair = pair_lookup.get((j_name, k_name))
                if pair is None:
                    pair = pair_lookup.get((k_name, j_name))
                if pair is None:
                    continue
                edge[j, k, :6] = bond_features2(pos, pair[0], pair[1], mol, rings)
                edge[j, k, 6] = j_index * 0.01
                edge[j, k, 7] = k_index * 0.01
                edge[k, j, :] = edge[j, k, :]

        dv.append(node)
        de.append(edge)
        dp.append(proximity)
        dy.append(target)
        dsmi.append(smi)

    dv = sparse.COO.from_numpy(np.asarray(dv, dtype=np.int8))
    de = sparse.COO.from_numpy(np.asarray(de, dtype=np.float32))
    dp = np.asarray(dp, dtype=np.float32)
    dy = np.asarray(dy, dtype=np.float64)
    dsmi = np.asarray(dsmi)

    with out_path.open("wb") as fw:
        pkl.dump([dv, de, dp, dy, dsmi], fw)

    if rdkit_sdf_path != sdf_path and rdkit_sdf_path.exists():
        try:
            shutil.copyfile(rdkit_sdf_path, sdf_path)
        except OSError:
            pass

    return {
        "output_pkl": out_path,
        "sdf_path": rdkit_sdf_path,
        "sample_count": len(dy),
        "skipped_empty_graph_count": len(skipped_empty_graphs),
        "skipped_empty_graphs": skipped_empty_graphs,
        "shapes": {
            "DV": dv.shape,
            "DE": de.shape,
            "DP": dp.shape,
            "DY": dy.shape,
        },
    }


def run_prepare(input_csv: Path, output_pkl: Path, work_csv: Path | None = None, sdf_path: Path | None = None, limit: int | None = None):
    input_csv = Path(input_csv).expanduser().resolve()
    output_pkl = Path(output_pkl).expanduser().resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"Missing input csv for PU-MPNN prepare: {input_csv}")

    work_csv = Path(work_csv).expanduser().resolve() if work_csv else output_pkl.with_name("release_check.csv")
    sdf_path = (
        Path(sdf_path).expanduser().resolve()
        if sdf_path
        else Path(tempfile.gettempdir()) / f"{output_pkl.stem}_release_check.sdf"
    )

    build_release_check_csv(input_csv, work_csv, limit=limit)
    result = build_graph_pkl(work_csv, sdf_path, output_pkl)
    return {
        "input_csv": input_csv,
        "release_check_csv": work_csv,
        "sdf_path": sdf_path,
        **result,
    }


def build_parser():
    parser = argparse.ArgumentParser(description="Prepare unified PU-MPNN preprocessing outputs.")
    parser.add_argument("--input-csv", required=False)
    parser.add_argument("--output-pkl", required=False)
    parser.add_argument("--work-csv", default=None)
    parser.add_argument("--sdf-path", default=None)
    parser.add_argument("--limit", type=int, default=None)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[4]
    input_csv = Path(args.input_csv).expanduser() if args.input_csv else repo_root / "examples" / "pugraph_demo" / "input.csv"
    output_pkl = Path(args.output_pkl).expanduser() if args.output_pkl else repo_root / "output" / "pu_mpnn_prepare_demo" / "genwl3.pkl"
    work_csv = Path(args.work_csv).expanduser() if args.work_csv else output_pkl.with_name("release_check.csv")
    sdf_path = Path(args.sdf_path).expanduser() if args.sdf_path else output_pkl.with_name("release_check.sdf")

    result = run_prepare(input_csv=input_csv, output_pkl=output_pkl, work_csv=work_csv, sdf_path=sdf_path, limit=args.limit)
    print("saved", result["output_pkl"].as_posix())
    print("release_check_csv", result["release_check_csv"].as_posix())
    print("sdf_path", result["sdf_path"].as_posix())
    print("shapes", result["shapes"]["DV"], result["shapes"]["DE"], result["shapes"]["DP"], result["shapes"]["DY"])


if __name__ == "__main__":
    main()
