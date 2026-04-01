"""
ShapeFBT capability experiment: approximate a fixed XGBoost target while varying only
n_estimators. No random hyperparameter sampling.

Fixed configuration (see make_xgb_classifier / make_shapefbt):
  xgb_max_depth=6; learning_rate=0.1; subsample=1.0; colsample_bytree=1.0; (+ other pinned XGB params)
  max_number_of_conjunctions=500; min_samples_split=10; outer/inner depth=6

Metrics:
  - % recovered: agreement between ShapeFBT class predictions and XGBoost predictions
  - reconstruction_error: 1 - agreement rate
  - runtime: XGBoost fit time and ShapeFBT fit time

Usage:
  python shapefbt_capability_test.py --folds 0 --home-dir ..
  python shapefbt_capability_test.py --datasets magic raisin bidding --n-estimators 50
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

_repo_root = Path(__file__).resolve().parent.parent
_tests_dir = Path(__file__).resolve().parent
# Repo root: ShapeFBT; tests/: data_utils (same pattern as shapefbt_run_new.py)
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_tests_dir))

from ShapeFBT import ShapeFBT  # noqa: E402
from data_utils import DataFactory_clf  # noqa: E402


def build_train_df(X: np.ndarray, y: np.ndarray, label_col: str = "y"):
    n_features = X.shape[1]
    feature_cols = [f"f{i}" for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df[label_col] = y
    return train_df, feature_cols, label_col


def xgb_objective_for_y(y: np.ndarray) -> str:
    n_class = int(np.unique(y).size)
    return "multi:softprob" if n_class > 2 else "binary:logitraw"


def make_xgb_classifier(
    *,
    n_estimators: int,
    max_depth: int,
    random_state: int,
    y_train: np.ndarray,
) -> XGBClassifier:
    """Pinned XGBoost target; only n_estimators (and optional CLI max_depth) vary."""
    return XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        colsample_bytree=1.0,
        subsample=1.0,
        min_child_weight=1,
        reg_lambda=0.0,
        reg_alpha=0.0,
        gamma=0.0,
        random_state=random_state,
        objective=xgb_objective_for_y(y_train),
        eval_metric="logloss",
        n_jobs=1,
        verbosity=0,
    )


def make_shapefbt() -> ShapeFBT:
    """Pinned ShapeFBT approximator (no random search)."""
    return ShapeFBT(
        outer_tree_max_depth=6,
        inner_tree_max_depth=6,
        min_forest_size=5,
        max_number_of_conjunctions=500,
        pruning_method=None,
        min_samples_split=20,
        min_conjunctions_split=2,
        min_impurity_decrease=0.0,
        k=2,
        inner_tree_criterion="entropy",
        verbose=False,
    )


def run_one(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    n_estimators: int,
    xgb_max_depth: int,
    random_state: int,
    feature_dict,
):
    train_df, feature_cols, label_col = build_train_df(X_train, y_train)
    val_df, _, _ = build_train_df(X_val, y_val, label_col=label_col)
    test_df, _, _ = build_train_df(X_test, y_test, label_col=label_col)

    xgb_model = make_xgb_classifier(
        n_estimators=n_estimators,
        max_depth=xgb_max_depth,
        random_state=random_state,
        y_train=y_train,
    )

    t0 = time.perf_counter()
    xgb_model.fit(X_train, y_train)
    xgb_fit_s = time.perf_counter() - t0

    model = make_shapefbt()
    t1 = time.perf_counter()
    model.fit(
        train_df,
        feature_cols=feature_cols,
        label_col=label_col,
        xgb_model=xgb_model,
        feature_dict=feature_dict,
    )
    shape_fit_s = time.perf_counter() - t1

    xgb_train = xgb_model.predict(X_train)
    xgb_val = xgb_model.predict(X_val)
    xgb_test = xgb_model.predict(X_test)

    shape_train, _ = model.predict_Xy(train_df[feature_cols])
    shape_val, _ = model.predict_Xy(val_df[feature_cols])
    shape_test, _ = model.predict_Xy(test_df[feature_cols])

    def recovered_pct(a, b):
        return float(100.0 * np.mean(a == b))

    out = {
        "pct_recovered_train": recovered_pct(shape_train, xgb_train),
        "pct_recovered_val": recovered_pct(shape_val, xgb_val),
        "pct_recovered_test": recovered_pct(shape_test, xgb_test),
        "reconstruction_error_train": 1.0 - float(np.mean(shape_train == xgb_train)),
        "reconstruction_error_val": 1.0 - float(np.mean(shape_val == xgb_val)),
        "reconstruction_error_test": 1.0 - float(np.mean(shape_test == xgb_test)),
        "xgb_acc_train": accuracy_score(y_train, xgb_train),
        "xgb_acc_val": accuracy_score(y_val, xgb_val),
        "xgb_acc_test": accuracy_score(y_test, xgb_test),
        "shape_acc_train": accuracy_score(y_train, shape_train),
        "shape_acc_val": accuracy_score(y_val, shape_val),
        "shape_acc_test": accuracy_score(y_test, shape_test),
        "time_xgb_fit_s": xgb_fit_s,
        "time_shapefbt_fit_s": shape_fit_s,
        "time_total_fit_s": xgb_fit_s + shape_fit_s,
    }
    return out


def main():
    parser = argparse.ArgumentParser(
        description="ShapeFBT vs fixed 'wild' XGBoost: reconstruction vs n_estimators (capability test)."
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["magic", "raisin", "bidding"],
        help="Datasets to run (DataFactory_clf names). Default: magic raisin bidding.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="Fold indices (same convention as DataFactory_clf.get_data).",
    )
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument(
        "--xgb-max-depth",
        type=int,
        default=6,
        help="Tree depth for the target XGBoost model (default 6 for this experiment).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=[50, 100, 250, 500],
        help="Independent variable: boosting rounds for the target XGB.",
    )
    parser.add_argument(
        "--home-dir",
        type=str,
        default=".",
        help="Project root; data cache at <home-dir>/data, results under <home-dir>/results unless overridden.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Path to CSV for append rows. Default: <home-dir>/results/capability/shapefbt_capability.csv",
    )
    args = parser.parse_args()

    home = os.path.abspath(args.home_dir)
    out_csv = args.output_csv or os.path.join(
        home, "results", "capability", "shapefbt_capability.csv"
    )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "dataset",
        "fold",
        "random_state",
        "n_estimators",
        "xgb_max_depth",
        "pct_recovered_train",
        "pct_recovered_val",
        "pct_recovered_test",
        "reconstruction_error_train",
        "reconstruction_error_val",
        "reconstruction_error_test",
        "xgb_acc_train",
        "xgb_acc_val",
        "xgb_acc_test",
        "shape_acc_train",
        "shape_acc_val",
        "shape_acc_test",
        "time_xgb_fit_s",
        "time_shapefbt_fit_s",
        "time_total_fit_s",
    ]

    write_header = not os.path.isfile(out_csv)
    with open(out_csv, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for dataset_name in args.datasets:
            data_factory = DataFactory_clf(
                dataset_name, cache_dir=os.path.join(home, "data")
            )
            feature_dict = getattr(data_factory, "feature_dict", None)

            for fold_idx in args.folds:
                X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(
                    fold_idx=fold_idx
                )
                random_state = args.base_seed + fold_idx

                for n_est in args.n_estimators:
                    metrics = run_one(
                        X_train=X_train,
                        y_train=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        X_test=X_test,
                        y_test=y_test,
                        n_estimators=n_est,
                        xgb_max_depth=args.xgb_max_depth,
                        random_state=random_state,
                        feature_dict=feature_dict,
                    )
                    row = {
                        "dataset": dataset_name,
                        "fold": fold_idx,
                        "random_state": random_state,
                        "n_estimators": n_est,
                        "xgb_max_depth": args.xgb_max_depth,
                        **{k: metrics[k] for k in metrics},
                    }
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"dataset={dataset_name} fold={fold_idx} n_estimators={n_est} "
                        f"pct_recovered_test={metrics['pct_recovered_test']:.2f}% "
                        f"shape_fit_s={metrics['time_shapefbt_fit_s']:.2f}"
                    )

    print(f"Wrote results to {out_csv}")


if __name__ == "__main__":
    main()
