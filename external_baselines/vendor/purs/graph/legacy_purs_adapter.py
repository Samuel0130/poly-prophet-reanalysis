"""Shared adapter for legacy graph workflows that still rely on old PURS-style feature builders."""

from __future__ import annotations

import csv
import os
import tempfile
from pathlib import Path

import pandas as pd
import torchgraphs as tg

from purs.graph.pu_gn_exp.polymer_unit import PURS as legacy_purs


def _normalize_rows(df: pd.DataFrame):
    rows = []
    for _, row in df[["Compound ID", "smiles"]].iterrows():
        rows.append(
            {
                "Compound ID": str(row["Compound ID"]),
                "smiles": str(row["smiles"]),
            }
        )
    return rows


def _write_legacy_csv(rows):
    with tempfile.NamedTemporaryFile("w", newline="", suffix=".csv", delete=False, encoding="utf-8") as tmp:
        writer = csv.writer(tmp)
        for row in rows:
            writer.writerow([row["Compound ID"], row["smiles"]])
        return tmp.name


def _build_graph_dist_for_rows(rows):
    if not rows:
        return {}

    tmp_path = _write_legacy_csv(rows)
    try:
        smi_list, _name_list0, name_list, mol_list, _num_list = legacy_purs.process_smiles(tmp_path)
    finally:
        os.unlink(tmp_path)

    ring_total_list, total_neighbor_data, total_inner_dist, _total_end_atom_pair = legacy_purs.get_pu(
        smi_list,
        name_list,
    )
    total_bratch_dist = legacy_purs.get_bratch_dist2(smi_list, name_list)
    total_neighbor_data, total_inner_dist, ring_total_list = legacy_purs.update_bratch(
        name_list,
        smi_list,
        total_neighbor_data,
        total_inner_dist,
        total_bratch_dist,
    )
    total_neighbor_data, total_inner_dist, ring_total_list = legacy_purs.get_new_neighbor_data(
        total_neighbor_data,
        total_inner_dist,
        name_list,
        smi_list,
    )
    matrix_list = legacy_purs.get_adj(ring_total_list, total_neighbor_data, name_list)
    pu_index = legacy_purs.get_pu_dict(total_neighbor_data, ring_total_list)
    pair_atom_dist = legacy_purs.get_pair_atom(total_neighbor_data, total_inner_dist)
    maccs_dict = legacy_purs.get_MACCS(ring_total_list, False)
    edge_list, edge_num_list, senders_list, receivers_list, node_feature_dist = legacy_purs.get_feature(
        mol_list,
        name_list,
        matrix_list,
        pu_index,
        pair_atom_dist,
        total_inner_dist,
        maccs_dict,
    )

    graph_dist = {}
    for idx, name in enumerate(name_list):
        num_edges = edge_num_list[idx]
        node_features = node_feature_dist[name]
        num_nodes = len(node_features)
        edge_features = edge_list[idx]
        senders = senders_list[name]
        receivers = receivers_list[name]
        graph_dist[name] = tg.Graph(
            num_nodes=num_nodes,
            num_edges=num_edges,
            node_features=node_features,
            edge_features=edge_features,
            senders=senders,
            receivers=receivers,
        )

    return graph_dist


def _write_skip_report(source_path, skipped_rows):
    if not skipped_rows:
        return None
    source_path = Path(source_path)
    report_path = source_path.with_name(f"{source_path.stem}_legacy_graph_skipped.csv")
    pd.DataFrame(skipped_rows).to_csv(report_path, index=False)
    return report_path


def build_graph_dist_from_csv(file_name):
    """Build torchgraphs graph objects from a normalized graph input csv.

    The current graph baselines still depend on a subset of the legacy PURS
    feature-building logic. This adapter centralizes that dependency so
    downstream code no longer imports multiple embedded PURS variants directly.

    When the legacy bulk builder rejects one or more molecules, we fall back to
    per-row processing so a few graph-incompatible samples do not abort an
    otherwise valid large-batch run.
    """

    df = pd.read_csv(file_name)
    rows = _normalize_rows(df)

    try:
        return _build_graph_dist_for_rows(rows)
    except Exception as bulk_exc:
        graph_dist = {}
        skipped_rows = []
        for row in rows:
            try:
                graph_dist.update(_build_graph_dist_for_rows([row]))
            except Exception as row_exc:
                skipped_rows.append(
                    {
                        "Compound ID": row["Compound ID"],
                        "smiles": row["smiles"],
                        "error_type": type(row_exc).__name__,
                        "error_message": str(row_exc),
                    }
                )

        report_path = _write_skip_report(file_name, skipped_rows)
        if not graph_dist:
            raise RuntimeError(
                f"Legacy graph construction failed for every row in {file_name}. "
                f"First bulk error: {type(bulk_exc).__name__}: {bulk_exc}"
            ) from bulk_exc

        message = (
            f"[legacy_purs_adapter] bulk graph build failed for {file_name}; "
            f"recovered {len(graph_dist)} graphs with per-row fallback and skipped {len(skipped_rows)} rows. "
            f"First bulk error: {type(bulk_exc).__name__}: {bulk_exc}"
        )
        if report_path is not None:
            message += f" Skip report: {report_path}"
        print(message)
        return graph_dist
