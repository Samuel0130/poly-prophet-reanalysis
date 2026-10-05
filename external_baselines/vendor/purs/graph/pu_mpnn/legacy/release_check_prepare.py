#!/usr/bin/env python
from pathlib import Path
import tempfile

import pandas as pd
from purs.graph.pu_mpnn.prepare import build_graph_pkl, build_release_check_csv


def main():
    repo_root = Path(__file__).resolve().parents[2]
    src_csv = repo_root / "test.csv"
    work_csv = Path(__file__).resolve().parent / "release_check.csv"
    work_sdf = Path(tempfile.gettempdir()) / "purs_release_check.sdf"
    out_pkl = Path(__file__).resolve().parent / "DATA" / "genwl3.pkl"

    build_release_check_csv(src_csv, work_csv)
    result = build_graph_pkl(work_csv, work_sdf, out_pkl)
    print("saved", result["output_pkl"].as_posix())
    print("shapes", result["shapes"]["DV"], result["shapes"]["DE"], result["shapes"]["DP"], result["shapes"]["DY"])


if __name__ == "__main__":
    main()
