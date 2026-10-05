"""Graph workflow build/train entry points."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from purs.core.recognize import recognize_units
from purs.graph.dataset import build_graph_input_csv


def _load_pu_gn_exp_template():
    template_path = Path(__file__).resolve().parent / "pu_gn_exp" / "polymer_unit" / "train.yaml"
    template = yaml.safe_load(template_path.read_text(encoding="utf-8")) or {}
    return template


def _resolve_existing_path(path_value, label):
    if path_value is None:
        return None
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Configured {label} does not exist: {path}")
    return path


def _validate_graph_config(config_path, cfg):
    backend = cfg.get("backend", "pu_gn_exp")
    mode = cfg.get("mode", "command_only")
    data_csv_value = cfg.get("data_csv")
    if data_csv_value is None and backend == "pu_gn_exp":
        data_csv_value = cfg.get("session", {}).get("data", {}).get("path")
    data_csv = _resolve_existing_path(data_csv_value, "data_csv")

    if backend == "pu_gn_exp":
        if mode not in {"command_only", "execute"}:
            raise ValueError(f"Unsupported pu_gn_exp mode: {mode}")
    elif backend == "pu_mpnn":
        if mode not in {"command_only", "execute"}:
            raise ValueError("PU-MPNN supports command_only or execute mode.")
        if data_csv is None:
            raise ValueError(
                f"Graph config {config_path} must define data_csv when backend='pu_mpnn'."
            )
        if not cfg.get("output_pkl"):
            raise ValueError(
                f"Graph config {config_path} must define output_pkl when backend='pu_mpnn'."
            )
    else:
        raise ValueError(f"Unsupported graph backend: {backend}")

    return {
        "backend": backend,
        "mode": mode,
        "data_csv": data_csv,
        "output_pkl": Path(cfg["output_pkl"]).expanduser().resolve() if cfg.get("output_pkl") else None,
    }


def build_pugraph(input_csv, output_dir=".", name_column=None, smiles_column=None, strict=False):
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    core_dir = output_dir / "core"
    core_result = recognize_units(
        input_csv=input_csv,
        name_column=name_column,
        smiles_column=smiles_column,
        output_dir=core_dir,
        strict=strict,
    )
    skipped_ids = []
    if core_result["skipped_count"] > 0:
        skipped_df = pd.read_csv(core_result["skipped_records_csv"])
        if "name" in skipped_df.columns:
            skipped_ids = skipped_df["name"].dropna().astype(str).tolist()
    dataset_result = build_graph_input_csv(
        input_csv=input_csv,
        output_dir=output_dir,
        name_column=name_column,
        smiles_column=smiles_column,
        exclude_ids=skipped_ids,
    )

    pu_gn_exp_template = output_dir / "pu_gn_exp_train.yaml"
    pu_gn_exp_payload = _load_pu_gn_exp_template()
    pu_gn_exp_payload.update(
        {
            "name": "polymer_unit_graph_experiment",
            "backend": "pu_gn_exp",
            "mode": "command_only",
            "data_csv": str(dataset_result["graph_input_csv"]),
            "output_dir": str(output_dir / "runs" / "pu_gn_exp"),
            "notes": "Switch mode to execute after reviewing paths and hyperparameters.",
        }
    )
    session = pu_gn_exp_payload.setdefault("session", {})
    session["epochs"] = int(session.get("epochs", 1) if core_result["sample_count"] > 8 else 1)
    session["batch_size"] = min(max(core_result["sample_count"], 1), int(session.get("batch_size", 50)))
    session.setdefault("data", {})
    session["data"]["path"] = str(dataset_result["graph_input_csv"])
    session["data"]["train"] = 0.7 if core_result["sample_count"] > 2 else 0.5
    session["data"]["val"] = 1 - session["data"]["train"]
    session.setdefault("log", {})
    session["log"]["folder"] = str(output_dir / "runs" / "pu_gn_exp" / "logs")
    session["log"]["when"] = []
    session.setdefault("checkpoint", {})
    session["checkpoint"]["folder"] = str(output_dir / "runs" / "pu_gn_exp")
    session["checkpoint"]["when"] = ["last epoch"]
    pu_gn_exp_template.write_text(yaml.safe_dump(pu_gn_exp_payload, sort_keys=False), encoding="utf-8")

    pu_mpnn_template = output_dir / "pu_mpnn_train.yaml"
    pu_mpnn_payload = {
        "backend": "pu_mpnn",
        "mode": "command_only",
        "data_csv": str(dataset_result["graph_input_csv"]),
        "output_pkl": str(output_dir / "pu_mpnn" / "genwl3.pkl"),
        "output_dir": str(output_dir / "runs" / "pu_mpnn"),
        "notes": "Use the unified PU-MPNN prepare wrapper before training.",
    }
    pu_mpnn_template.write_text(yaml.safe_dump(pu_mpnn_payload, sort_keys=False), encoding="utf-8")

    build_manifest = {
        "input_csv": str(Path(input_csv).expanduser().resolve()),
        "output_dir": str(output_dir),
        "core_output_dir": str(core_dir),
        "graph_input_csv": str(dataset_result["graph_input_csv"]),
        "dataset_manifest_json": str(dataset_result["manifest_json"]),
        "pu_gn_exp_train_yaml": str(pu_gn_exp_template),
        "pu_mpnn_train_yaml": str(pu_mpnn_template),
        "processed_samples": core_result["sample_count"],
        "skipped_rows": core_result["skipped_count"],
        "target_column": dataset_result["target_column"],
    }
    build_manifest_path = output_dir / "graph_build_manifest.json"
    build_manifest_path.write_text(json.dumps(build_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "output_dir": output_dir,
        "core_output_dir": core_dir,
        "graph_input_csv": dataset_result["graph_input_csv"],
        "dataset_manifest_json": dataset_result["manifest_json"],
        "build_manifest_json": build_manifest_path,
        "pu_gn_exp_train_yaml": pu_gn_exp_template,
        "pu_mpnn_train_yaml": pu_mpnn_template,
        "sample_count": core_result["sample_count"],
        "skipped_count": core_result["skipped_count"],
        "target_column": dataset_result["target_column"],
    }


def train_pugraph(config_path):
    config_path = Path(config_path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Missing graph config: {config_path}")

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    validated = _validate_graph_config(config_path, cfg)
    backend = validated["backend"]
    mode = validated["mode"]

    if backend == "pu_gn_exp":
        package_dir = Path(__file__).resolve().parent / "pu_gn_exp"
        if mode == "command_only":
            command = [
                "python",
                "-m",
                "polymer_unit.train",
                "--experiment",
                str(config_path),
            ]
            return {
                "backend": backend,
                "mode": mode,
                "workdir": package_dir,
                "command": command,
                "executed": False,
            }
        if mode == "execute":
            python_executable = cfg.get("python_executable") or sys.executable or shutil.which("python")
            if not python_executable:
                raise RuntimeError("Could not find a Python executable for pu_gn_exp training.")
            command = [python_executable, "-m", "polymer_unit.train", "--experiment", str(config_path)]
            completed = subprocess.run(command, cwd=package_dir, check=True, capture_output=True, text=True)
            return {
                "backend": backend,
                "mode": mode,
                "workdir": package_dir,
                "command": command,
                "executed": True,
                "data_csv": str(validated["data_csv"]) if validated["data_csv"] else None,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

    if backend == "pu_mpnn":
        package_dir = Path(__file__).resolve().parents[3]
        output_pkl = validated["output_pkl"]
        work_csv = output_pkl.with_name("release_check.csv")
        sdf_path = output_pkl.with_suffix(".sdf")
        python_executable = cfg.get("python_executable") or sys.executable or shutil.which("python") or "python"
        command = [
            python_executable,
            "-m",
            "purs.graph.pu_mpnn.prepare",
            "--input-csv",
            str(validated["data_csv"]),
            "--output-pkl",
            str(output_pkl),
            "--work-csv",
            str(work_csv),
            "--sdf-path",
            str(sdf_path),
        ]
        if cfg.get("limit") is not None:
            command.extend(["--limit", str(cfg["limit"])])
        if mode == "execute":
            env = dict(os.environ)
            existing_pythonpath = env.get("PYTHONPATH")
            repo_src = str(package_dir / "src")
            env["PYTHONPATH"] = repo_src if not existing_pythonpath else f"{repo_src}{os.pathsep}{existing_pythonpath}"
            completed = subprocess.run(command, cwd=package_dir, check=True, capture_output=True, text=True, env=env)
            return {
                "backend": backend,
                "mode": mode,
                "workdir": package_dir,
                "command": command,
                "executed": True,
                "data_csv": str(validated["data_csv"]),
                "output_pkl": str(output_pkl),
                "release_check_csv": str(work_csv),
                "sdf_path": str(sdf_path),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        return {
            "backend": backend,
            "mode": mode,
            "workdir": package_dir,
            "command": command,
            "executed": False,
            "data_csv": str(validated["data_csv"]),
            "output_pkl": str(output_pkl),
            "release_check_csv": str(work_csv),
            "sdf_path": str(sdf_path),
            "notes": "Run the unified prepare wrapper first, then train the legacy TensorFlow model from src/purs/graph/pu_mpnn/legacy.",
        }
