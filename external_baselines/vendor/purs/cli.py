"""Command-line entry point for the unified PURS package."""

from __future__ import annotations

import argparse
from importlib import import_module


def _load_attr(module_name: str, attr_name: str):
    try:
        module = import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or module_name
        raise SystemExit(
            f"Missing optional dependency '{missing}' while loading '{module_name}'. "
            "Install the appropriate conda environment from environments/ or the matching extras "
            "before running this command."
        ) from exc
    return getattr(module, attr_name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="purs",
        description="Unified CLI for PURS, PUFp, and PUGraph workflows.",
    )
    subparsers = parser.add_subparsers(dest="command")

    recognize = subparsers.add_parser("recognize", help="Run polymer-unit recognition.")
    recognize.add_argument("--input-csv", required=True)
    recognize.add_argument("--name-column", default=None)
    recognize.add_argument("--smiles-column", default=None)
    recognize.add_argument("--output-dir", required=True)
    recognize.add_argument("--strict", action="store_true")
    recognize.set_defaults(handler=_handle_recognize)

    classify = subparsers.add_parser("classify", help="Run polymer-unit classification.")
    classify.add_argument("--ring-total-list", required=True)
    classify.add_argument("--index-data", required=True)
    classify.add_argument("--output-dir", required=True)
    classify.set_defaults(handler=_handle_classify)

    fingerprint = subparsers.add_parser("fingerprint", help="Build polymer-unit fingerprints.")
    fingerprint.add_argument("--input-csv", required=True)
    fingerprint.add_argument("--name-column", default=None)
    fingerprint.add_argument("--smiles-column", default=None)
    fingerprint.add_argument("--output-dir", required=True)
    fingerprint.add_argument("--strict", action="store_true")
    fingerprint.set_defaults(handler=_handle_fingerprint)

    ml = subparsers.add_parser("ml", help="Run traditional ML workflows.")
    ml_subparsers = ml.add_subparsers(dest="ml_command")
    for name in ("rf", "krr", "svm"):
        model_parser = ml_subparsers.add_parser(name, help=f"Run the {name.upper()} baseline.")
        model_parser.add_argument("--feature-csv", required=True)
        model_parser.add_argument("--target-csv", required=True)
        model_parser.add_argument("--id-column", required=True)
        model_parser.add_argument("--target-column", required=True)
        model_parser.add_argument("--test-size", type=float, default=0.2)
        model_parser.add_argument("--random-state", type=int, default=111)
        model_parser.add_argument("--cv", type=int, default=5)
        model_parser.add_argument("--n-jobs", type=int, default=-1)
        model_parser.add_argument("--output-plot", default=None)
        model_parser.add_argument("--quick", action="store_true")
        model_parser.set_defaults(handler=_handle_ml, model_name=name)

    graph = subparsers.add_parser("graph", help="Run graph construction or training workflows.")
    graph_subparsers = graph.add_subparsers(dest="graph_command")

    graph_build = graph_subparsers.add_parser("build", help="Build graph-ready datasets.")
    graph_build.add_argument("--input-csv", required=True)
    graph_build.add_argument("--name-column", default=None)
    graph_build.add_argument("--smiles-column", default=None)
    graph_build.add_argument("--output-dir", required=True)
    graph_build.set_defaults(handler=_handle_graph_build)

    graph_train = graph_subparsers.add_parser("train", help="Train a graph model from config.")
    graph_train.add_argument("--config", required=True)
    graph_train.set_defaults(handler=_handle_graph_train)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return
    handler(args)


def _handle_recognize(args: argparse.Namespace) -> None:
    recognize_units = _load_attr("purs.core.recognize", "recognize_units")
    result = recognize_units(
        input_csv=args.input_csv,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
        strict=args.strict,
    )
    print(f"Using name column: {result['name_column']}")
    print(f"Using smiles column: {result['smiles_column']}")
    print(f"Input rows: {result['total_rows']}")
    print(f"Processed samples: {result['sample_count']}")
    print(f"Skipped rows: {result['skipped_count']}")
    print(f"Recognized unique units: {result['unit_count']}")
    print(f"Skipped-record report: {result['skipped_records_csv']}")
    print(f"Output directory: {result['output_dir']}")


def _handle_classify(args: argparse.Namespace) -> None:
    classify_units = _load_attr("purs.core.classify", "classify_units")
    result = classify_units(
        ring_total_list_path=args.ring_total_list,
        index_data_path=args.index_data,
        output_dir=args.output_dir,
    )
    print(f"Classified units: {result['ring_count']}")
    print(f"Output directory: {result['output_dir']}")


def _handle_fingerprint(args: argparse.Namespace) -> None:
    build_pufp = _load_attr("purs.fingerprint.build", "build_pufp")
    result = build_pufp(
        input_csv=args.input_csv,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
        strict=args.strict,
    )
    print(f"Using name column: {result['name_column']}")
    print(f"Using smiles column: {result['smiles_column']}")
    print(f"Input rows: {result['total_rows']}")
    print(f"Processed samples: {result['sample_count']}")
    print(f"Skipped rows: {result['skipped_count']}")
    print(f"Recognized unique units: {result['unit_count']}")
    print(f"Classified units: {result['ring_count']}")
    print(f"Skipped-record report: {result['skipped_records_csv']}")
    print(f"Run summary: {result['run_summary_json']}")
    print(f"Output directory: {result['output_dir']}")


def _handle_ml(args: argparse.Namespace) -> None:
    if args.model_name == "rf":
        run_rf = _load_attr("purs.ml.rf", "run_rf")
    elif args.model_name == "krr":
        run_krr = _load_attr("purs.ml.krr", "run_krr")
    elif args.model_name == "svm":
        run_svm = _load_attr("purs.ml.svm", "run_svm")

    common_kwargs = {
        "feature_csv": args.feature_csv,
        "target_csv": args.target_csv,
        "id_column": args.id_column,
        "target_column": args.target_column,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "cv": args.cv,
        "n_jobs": args.n_jobs,
        "output_plot": args.output_plot,
        "quick": args.quick,
    }
    if args.model_name == "rf":
        result = run_rf(**common_kwargs)
    elif args.model_name == "krr":
        result = run_krr(**common_kwargs)
    elif args.model_name == "svm":
        result = run_svm(**common_kwargs)
    else:
        raise ValueError(f"Unsupported ml model: {args.model_name}")

    print(f"Model: {args.model_name}")
    print(f"Matched samples: {result['sample_count']}")
    print(f"Feature columns: {result['feature_count']}")
    print(f"Best params: {result['best_params']}")


def _handle_graph_build(args: argparse.Namespace) -> None:
    build_pugraph = _load_attr("purs.graph.builders", "build_pugraph")
    result = build_pugraph(
        input_csv=args.input_csv,
        name_column=args.name_column,
        smiles_column=args.smiles_column,
        output_dir=args.output_dir,
    )
    print(f"Processed samples: {result['sample_count']}")
    print(f"Skipped rows: {result['skipped_count']}")
    print(f"Graph input csv: {result['graph_input_csv']}")
    print(f"Dataset manifest: {result['dataset_manifest_json']}")
    print(f"Build manifest: {result['build_manifest_json']}")
    print(f"PU-gn-exp config template: {result['pu_gn_exp_train_yaml']}")
    print(f"PU-MPNN config template: {result['pu_mpnn_train_yaml']}")
    print(f"Output directory: {result['output_dir']}")


def _handle_graph_train(args: argparse.Namespace) -> None:
    train_pugraph = _load_attr("purs.graph.builders", "train_pugraph")
    result = train_pugraph(args.config)
    print(f"Backend: {result['backend']}")
    print(f"Mode: {result['mode']}")
    print(f"Working directory: {result['workdir']}")
    print(f"Command: {' '.join(map(str, result['command']))}")
    if result.get("notes"):
        print(f"Notes: {result['notes']}")
    if result.get("executed"):
        print("Execution completed.")


if __name__ == "__main__":
    main()
