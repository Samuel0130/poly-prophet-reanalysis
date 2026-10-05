from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def load_feature_class_target(
    feature_csv: str | Path,
    target_csv: str | Path,
    id_column: str,
    target_column: str,
    allowed_labels: list[str] | None = None,
):
    feature_path = Path(feature_csv)
    target_path = Path(target_csv)
    if not feature_path.exists():
        raise FileNotFoundError(f"Missing feature csv: {feature_csv}")
    if not target_path.exists():
        raise FileNotFoundError(f"Missing target csv: {target_csv}")

    feature_df = pd.read_csv(feature_path, index_col=0)
    target_df = pd.read_csv(target_path)

    if feature_df.shape[1] == 0:
        raise ValueError(
            "The feature table contains 0 columns. This usually means the upstream fingerprint "
            "step did not produce usable polymer-unit features for the chosen input data."
        )

    if id_column not in target_df.columns:
        raise KeyError(f"Missing id column in target csv: {id_column}")
    if target_column not in target_df.columns:
        raise KeyError(f"Missing target column in target csv: {target_column}")

    feature_df.index = feature_df.index.map(str)
    target_df[id_column] = target_df[id_column].map(str)
    target_df = target_df[target_df[target_column].notna()].copy()
    if allowed_labels is not None:
        target_df = target_df[target_df[target_column].isin(allowed_labels)].copy()

    merged = feature_df.merge(
        target_df[[id_column, target_column]],
        left_index=True,
        right_on=id_column,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No shared sample ids were found between the feature table and target table.")

    X = merged[feature_df.columns].to_numpy(dtype=float)
    y = merged[target_column].astype(str).to_numpy()
    ids = merged[id_column].tolist()
    return X, y, ids, feature_df.columns.tolist()


def split_classification_dataset(X, y, test_size=0.2, random_state=111):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def resolve_classification_cv(cv: int, y_train) -> int:
    label_counts = pd.Series(y_train).value_counts()
    if len(label_counts) < 2:
        raise ValueError("Classification requires at least two classes in the training split.")
    min_count = int(label_counts.min())
    if min_count < 2:
        raise ValueError(
            "At least one class has fewer than 2 training samples, so cross-validation cannot run."
        )
    return min(cv, min_count)


def classification_metrics(y_true, y_pred, labels: list[str] | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "confusion_matrix": cm.tolist(),
        "labels": labels or sorted(pd.unique(np.concatenate([y_true, y_pred])).tolist()),
    }


def print_classification_metrics(label, metrics):
    print(
        f"{label} -> "
        f"accuracy: {metrics['accuracy']:.6f}  "
        f"balanced_accuracy: {metrics['balanced_accuracy']:.6f}  "
        f"macro_f1: {metrics['macro_f1']:.6f}"
    )


def _run_classifier(
    estimator,
    param_grid,
    *,
    feature_csv,
    target_csv,
    id_column,
    target_column,
    allowed_labels=None,
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
):
    X, y, ids, feature_names = load_feature_class_target(
        feature_csv=feature_csv,
        target_csv=target_csv,
        id_column=id_column,
        target_column=target_column,
        allowed_labels=allowed_labels,
    )
    print("Feature columns:", len(feature_names))
    print("Matched samples:", len(y))
    print("Class distribution:", pd.Series(y).value_counts().to_dict())

    X_train, X_test, y_train, y_test = split_classification_dataset(
        X, y, test_size=test_size, random_state=random_state
    )
    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    effective_cv = resolve_classification_cv(cv, y_train)

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        cv=effective_cv,
        n_jobs=n_jobs,
        scoring="accuracy",
    )
    gs = gs.fit(X_train, y_train)
    print("Best params:", gs.best_params_)

    y_pred_train = gs.predict(X_train)
    y_pred_test = gs.predict(X_test)
    labels = allowed_labels or sorted(pd.Series(y).value_counts().index.tolist())
    train_metrics = classification_metrics(y_train, y_pred_train, labels=labels)
    test_metrics = classification_metrics(y_test, y_pred_test, labels=labels)
    print_classification_metrics("Train", train_metrics)
    print_classification_metrics("Test", test_metrics)

    return {
        "feature_count": len(feature_names),
        "sample_count": len(y),
        "class_distribution": pd.Series(y).value_counts().to_dict(),
        "best_params": gs.best_params_,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
    }


def run_rf_classifier(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="label",
    allowed_labels=None,
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
    quick=False,
):
    if quick:
        param_grid = [{
            "max_depth": [6, 10],
            "max_features": ["sqrt"],
            "n_estimators": [50, 100],
        }]
    else:
        param_grid = [{
            "max_depth": [4, 6, 10, 20],
            "max_features": ["sqrt", "log2"],
            "n_estimators": [50, 100, 200],
        }]
    return _run_classifier(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        feature_csv=feature_csv,
        target_csv=target_csv,
        id_column=id_column,
        target_column=target_column,
        allowed_labels=allowed_labels,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        n_jobs=n_jobs,
    )


def run_knn_classifier(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="label",
    allowed_labels=None,
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
    quick=False,
):
    if quick:
        param_grid = [{
            "knn__n_neighbors": [3, 5, 7],
            "knn__weights": ["uniform", "distance"],
        }]
    else:
        param_grid = [{
            "knn__n_neighbors": [3, 5, 7, 9, 11],
            "knn__weights": ["uniform", "distance"],
        }]
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])
    return _run_classifier(
        estimator,
        param_grid,
        feature_csv=feature_csv,
        target_csv=target_csv,
        id_column=id_column,
        target_column=target_column,
        allowed_labels=allowed_labels,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        n_jobs=n_jobs,
    )


def run_svm_classifier(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="label",
    allowed_labels=None,
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
    quick=False,
):
    if quick:
        param_grid = [{
            "svc__C": [1, 2, 10],
            "svc__gamma": [1e-4, 1e-3],
        }]
    else:
        param_grid = [{
            "svc__C": [0.5, 1, 2, 10, 50],
            "svc__gamma": [1e-5, 1e-4, 1e-3, 1e-2],
        }]
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf", random_state=random_state)),
    ])
    return _run_classifier(
        estimator,
        param_grid,
        feature_csv=feature_csv,
        target_csv=target_csv,
        id_column=id_column,
        target_column=target_column,
        allowed_labels=allowed_labels,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        n_jobs=n_jobs,
    )


def run_mlp_classifier(
    feature_csv="output_n/number.csv",
    target_csv="OPV_exp_data.csv",
    id_column="No.",
    target_column="label",
    allowed_labels=None,
    test_size=0.2,
    random_state=111,
    cv=5,
    n_jobs=-1,
    quick=False,
):
    if quick:
        param_grid = [{
            "mlp__hidden_layer_sizes": [(64,), (128,)],
            "mlp__alpha": [1e-4, 1e-3],
        }]
    else:
        param_grid = [{
            "mlp__hidden_layer_sizes": [(64,), (128,), (128, 64)],
            "mlp__alpha": [1e-5, 1e-4, 1e-3],
        }]
    estimator = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=1000, random_state=random_state)),
    ])
    return _run_classifier(
        estimator,
        param_grid,
        feature_csv=feature_csv,
        target_csv=target_csv,
        id_column=id_column,
        target_column=target_column,
        allowed_labels=allowed_labels,
        test_size=test_size,
        random_state=random_state,
        cv=cv,
        n_jobs=n_jobs,
    )


def main(**kwargs):
    run_rf_classifier(**kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a classifier on PUFp features.")
    parser.add_argument("--feature-csv", default="output_n/number.csv")
    parser.add_argument("--target-csv", default="OPV_exp_data.csv")
    parser.add_argument("--id-column", default="No.")
    parser.add_argument("--target-column", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=111)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    main(
        feature_csv=args.feature_csv,
        target_csv=args.target_csv,
        id_column=args.id_column,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
        cv=args.cv,
        n_jobs=args.n_jobs,
        quick=args.quick,
    )
