#!/usr/bin/env python
# coding: utf-8

import argparse
import csv
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from . import structure_identity as F


def normalize_smiles(smiles):
    return smiles.replace("/", "").replace("\\", "")


@contextmanager
def silence_rdkit_messages():
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        yield
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")


def parse_smiles_safely(smiles):
    normalized_smiles = normalize_smiles(smiles)
    with silence_rdkit_messages():
        mol = Chem.MolFromSmiles(normalized_smiles)
    if not mol:
        return None, normalized_smiles, "rdkit_parse_failed"
    return mol, normalized_smiles, None


def detect_name_column(fieldnames):
    for candidate in ["name", "Name", "No.", "No", "ID", "Compound ID"]:
        if candidate in fieldnames:
            return candidate
    return fieldnames[0]


def detect_smiles_column(fieldnames):
    candidates = [col for col in ["smiles", "SMILES", "n(SMILES)", "p(SMILES)"] if col in fieldnames]
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "Multiple SMILES columns detected. Please specify one with --smiles-column, "
            f"for example: {candidates}"
        )
    raise ValueError("No supported SMILES column found in the input csv.")


def load_input_records(csv_path, name_column=None, smiles_column=None):
    smi_list = []
    name_list = []
    name_counts = Counter()
    skipped_records = []
    total_rows = 0

    with open(csv_path, newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("Input csv does not contain a readable header row.")

        if name_column is None:
            name_column = detect_name_column(reader.fieldnames)
        if smiles_column is None:
            smiles_column = detect_smiles_column(reader.fieldnames)

        for row in reader:
            total_rows += 1
            name = str(row.get(name_column, "")).strip()
            smiles = str(row.get(smiles_column, "")).strip()
            if not name:
                skipped_records.append(
                    {
                        "row_number": total_rows,
                        "name": "",
                        "smiles": smiles,
                        "normalized_smiles": "",
                        "reason": "missing_name",
                    }
                )
                continue
            if not smiles:
                skipped_records.append(
                    {
                        "row_number": total_rows,
                        "name": name,
                        "smiles": "",
                        "normalized_smiles": "",
                        "reason": "missing_smiles",
                    }
                )
                continue

            mol, normalized_smiles, reason = parse_smiles_safely(smiles)
            if not mol:
                skipped_records.append(
                    {
                        "row_number": total_rows,
                        "name": name,
                        "smiles": smiles,
                        "normalized_smiles": normalized_smiles,
                        "reason": reason,
                    }
                )
                continue

            smi_list.append(Chem.MolToSmiles(mol))
            duplicate_id = name_counts[name]
            if duplicate_id:
                name_list.append(f"{name}-{duplicate_id}")
            else:
                name_list.append(name)
            name_counts[name] += 1

    return smi_list, name_list, name_column, smiles_column, skipped_records, total_rows


def build_neighbor_data(smi_list, name_list):
    ring_total_list = []
    total_neighbor_data = {}

    for idx, smiles in enumerate(smi_list):
        name = name_list[idx]

        left_index_list, right_index_list, index_list = F.get_bracket_index(smiles)
        cp_list = F.pairing(smiles, index_list, left_index_list, right_index_list)
        index_arr = np.array(index_list)
        smallest_r = F.smallest(cp_list, index_arr)
        str_df = F.structure_DataFrame(cp_list, smallest_r, right_index_list, left_index_list)

        independent_cp, dependent_cp, bratch_cp, bratch = F.rigin_type_classify(cp_list, smiles, smallest_r, str_df)
        cp_data = F.get_cp_data(cp_list, smallest_r, str_df, independent_cp, bratch_cp)
        _, index_data, index_cp, _ = F.find_independent_str(
            smiles, smallest_r, cp_data, independent_cp, dependent_cp, bratch_cp
        )

        br = {}
        index_data2 = {}
        for key, value in index_data.items():
            unit_smiles = value[1]
            unit_smiles = F.add_bracket(unit_smiles)
            unit_smiles = F.make_smi(unit_smiles)
            unit_smiles = F.link_c(unit_smiles)
            unit_smiles, branch_prefix = F.bratch_in_string(unit_smiles)
            mol, _, _ = parse_smiles_safely(unit_smiles)
            if mol:
                unit_smiles = Chem.MolToSmiles(mol)
            unit_smiles, branch_suffix = F.bratch_in_string(unit_smiles)
            br[key] = branch_prefix + branch_suffix
            index_data2[key] = [value[0], unit_smiles]

        index_data3, index_cp2, br2 = F.make_con(index_data2, index_cp, br)
        index_data4 = F.delete_free_radical_in_index_data(index_data3)
        br3 = F.bratch_amend(br2)

        for _, value in index_data4.items():
            ring_total_list.append(value[1])

        for key, branches in br3.items():
            canonical_branches = []
            for branch in branches:
                mol, _, _ = parse_smiles_safely(branch)
                if mol:
                    smiles_branch = Chem.MolToSmiles(mol)
                    canonical_branches.append(smiles_branch)
                    ring_total_list.append(smiles_branch)
            br3[key] = canonical_branches

        neighbor_data = F.found_neighbor(br3, str_df, index_data3, index_cp2)
        neighbor_data2 = F.found_end_point_neighbour(smiles, neighbor_data, index_data3)

        for _, value in neighbor_data2.items():
            for neighbor_key, neighbor_value in value["right_neighbor"].items():
                if "[C]" in neighbor_value:
                    value["right_neighbor"][neighbor_key] = neighbor_value.replace("[C]", "C")
            for neighbor_key, neighbor_value in value["left_neighbor"].items():
                if "[C]" in neighbor_value:
                    value["left_neighbor"][neighbor_key] = neighbor_value.replace("[C]", "C")
            if "[C]" in value["self"]:
                value["self"] = value["self"].replace("[C]", "C")

        total_neighbor_data[name] = neighbor_data2

    ring_total_list2 = list(dict.fromkeys(ring_total_list))
    return ring_total_list2, total_neighbor_data


def write_ring_list(ring_total_list, output_dir):
    pd.DataFrame(np.array(ring_total_list)).to_csv(output_dir / "ring_total_list.csv")


def write_skipped_records(skipped_records, output_dir):
    output_path = output_dir / "skipped_records.csv"
    columns = ["row_number", "name", "smiles", "normalized_smiles", "reason"]
    pd.DataFrame(skipped_records, columns=columns).to_csv(output_path, index=False)
    return output_path


def write_run_summary(output_dir, summary):
    output_path = output_dir / "run_summary.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def build_fingerprint_tables(ring_total_list, total_neighbor_data, name_list, output_dir):
    ring_index = {ring: idx for idx, ring in enumerate(ring_total_list)}
    max_nodes = max((len(v) for v in total_neighbor_data.values()), default=0)
    long = len(ring_total_list)

    one_hot_rows = []
    number_rows = []
    adjacent_rows = []
    node_rows = []
    index_rows = []

    for name in name_list:
        data = total_neighbor_data[name]

        one_hot = np.zeros((1, long))
        counts = np.zeros((1, long))
        node_matrix = np.zeros((long, max_nodes))
        adjacent_matrix = np.zeros((max_nodes, max_nodes))
        index_vector = np.full(max_nodes, "none", dtype=object)

        self_list = list(data.keys())
        node_to_index = {node_name: idx for idx, node_name in enumerate(self_list)}

        for j, node_name in enumerate(self_list):
            unit_smiles = data[node_name]["self"]
            column = ring_index[unit_smiles]
            one_hot[:, column] = 1
            counts[:, column] += 1
            node_matrix[column, j] = 1
            index_vector[j] = column

        for j, node_name in enumerate(self_list):
            data2 = data[node_name]
            for neighbor_name in data2["right_neighbor"].keys():
                adjacent_matrix[j, node_to_index[neighbor_name]] = 1
            for neighbor_name in data2["left_neighbor"].keys():
                adjacent_matrix[j, node_to_index[neighbor_name]] = 1

        one_hot_rows.append(one_hot.tolist()[0])
        number_rows.append(counts.tolist()[0])
        adjacent_rows.append(adjacent_matrix.flatten())
        node_rows.append(node_matrix.flatten())
        index_rows.append(index_vector)

    pd.DataFrame(np.array(one_hot_rows), index=name_list).to_csv(output_dir / "one_hot.csv")
    pd.DataFrame(np.array(number_rows), index=name_list).to_csv(output_dir / "number.csv")
    pd.DataFrame(np.array(adjacent_rows), index=name_list).to_csv(output_dir / "adjacent_matrix.csv")
    pd.DataFrame(np.array(node_rows), index=name_list).to_csv(output_dir / "node_matrix.csv")
    pd.DataFrame(index_rows, index=name_list).to_csv(output_dir / "index_data.csv")


def recognize_units(input_csv="OPV_exp_data.csv", name_column=None, smiles_column=None, output_dir=".", strict=False):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    smi_list, name_list, resolved_name_column, resolved_smiles_column, skipped_records, total_rows = load_input_records(
        input_csv, name_column=name_column, smiles_column=smiles_column
    )
    skipped_records_path = write_skipped_records(skipped_records, output_dir)
    if strict and skipped_records:
        summary = {
            "input_csv": str(Path(input_csv).expanduser().resolve()),
            "name_column": resolved_name_column,
            "smiles_column": resolved_smiles_column,
            "total_rows": total_rows,
            "processed_samples": len(name_list),
            "skipped_rows": len(skipped_records),
            "recognized_unique_units": 0,
            "strict_mode": True,
            "status": "failed_due_to_skipped_rows",
            "skipped_records_csv": str(skipped_records_path),
        }
        write_run_summary(output_dir, summary)
        raise ValueError(
            f"Strict mode is enabled and {len(skipped_records)} rows were skipped. "
            f"See {skipped_records_path} for details."
        )
    ring_total_list, total_neighbor_data = build_neighbor_data(smi_list, name_list)
    write_ring_list(ring_total_list, output_dir)
    build_fingerprint_tables(ring_total_list, total_neighbor_data, name_list, output_dir)
    summary = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "name_column": resolved_name_column,
        "smiles_column": resolved_smiles_column,
        "total_rows": total_rows,
        "processed_samples": len(name_list),
        "skipped_rows": len(skipped_records),
        "recognized_unique_units": len(ring_total_list),
        "strict_mode": bool(strict),
        "status": "completed",
        "skipped_records_csv": str(skipped_records_path),
    }
    summary_path = write_run_summary(output_dir, summary)
    return {
        "output_dir": output_dir,
        "name_column": resolved_name_column,
        "smiles_column": resolved_smiles_column,
        "sample_count": len(name_list),
        "unit_count": len(ring_total_list),
        "total_rows": total_rows,
        "skipped_count": len(skipped_records),
        "skipped_records_csv": skipped_records_path,
        "run_summary_json": summary_path,
    }


def main(input_csv="OPV_exp_data.csv", name_column=None, smiles_column=None, output_dir=".", strict=False):
    result = recognize_units(
        input_csv=input_csv,
        name_column=name_column,
        smiles_column=smiles_column,
        output_dir=output_dir,
        strict=strict,
    )
    print(f"Using name column: {result['name_column']}")
    print(f"Using smiles column: {result['smiles_column']}")
    print(f"Input rows: {result['total_rows']}")
    print(f"Processed samples: {result['sample_count']}")
    print(f"Skipped rows: {result['skipped_count']}")
    print(f"Recognized unique units: {result['unit_count']}")
    print(f"Skipped-record report: {result['skipped_records_csv']}")
    print(f"Output directory: {result['output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate polymer-unit fingerprints (PUFp) from an input csv.")
    parser.add_argument("input_csv", nargs="?", default="OPV_exp_data.csv", help="Input csv file.")
    parser.add_argument("--name-column", default=None, help="Column used as the sample identifier.")
    parser.add_argument("--smiles-column", default=None, help="Column used as the SMILES source.")
    parser.add_argument("--output-dir", default=".", help="Directory used to store generated csv files.")
    parser.add_argument("--strict", action="store_true", help="Fail if any rows are skipped during parsing.")
    args = parser.parse_args()
    main(
        args.input_csv,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
        strict=args.strict,
    )
