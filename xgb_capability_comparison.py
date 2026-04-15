#!/usr/bin/env python3
"""
Fit XGBoost with the same pinned hyperparameters as tests/shapefbt_capability_run.py
across datasets / folds / (n_estimators, max_depth), and log structure + accuracy metrics
to a CSV for later analysis in a notebook.

Example:
  python xgb_capability_comparison.py \\
    --home-dir . \\
    --datasets magic raisin bidding \\
    --folds 0 1 2 3 4 \\
    --n-estimators 10 20 40 60 80 100 \\
    --max-depth 3 4 5 \\
    --trial-id 0
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

# Project root = directory containing this file
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "tests"))

from data_utils import DataFactory_clf  # noqa: E402


def _make_xgb(
    n_estimators: int,
    max_depth: int,
    random_state: int,
    n_jobs: int,
    device: str,
) -> XGBClassifier:
    """Match pinned settings from shapefbt_capability_run.py (XGBoost stage only)."""
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
        n_jobs=n_jobs,
        device=device,
        eval_metric="mlogloss",
        verbosity=0,
    )


def forest_structure_metrics(model: XGBClassifier) -> Dict[str, Any]:
    """
    Summarize the fitted booster using trees_to_dataframe() (XGBoost 1.3+).
    Depth is computed by walking Yes/No links from each tree root (no Depth column in 2.x dumps).
    """
    booster = model.get_booster()
    tdf = booster.trees_to_dataframe()
    tdf.columns = [c.lower() for c in tdf.columns]

    feat_col = "feature"
    tree_col = "tree"
    id_col = "id"

    n_trees = int(tdf[tree_col].nunique())
    per_tree_leaves: List[int] = []
    per_tree_depth: List[int] = []
    split_features: List[str] = []

    for _tid, g in tdf.groupby(tree_col):
        g = g.copy()
        g.index = g[id_col]
        leaves = int((g[feat_col] == "Leaf").sum())
        per_tree_leaves.append(leaves)

        root = g.iloc[0][id_col]

        def max_depth_from(node_id: str, depth: int) -> int:
            row = g.loc[node_id]
            if row[feat_col] == "Leaf":
                return depth
            yes, no = row["yes"], row["no"]
            outs: List[int] = []
            if pd.notna(yes):
                outs.append(max_depth_from(yes, depth + 1))
            if pd.notna(no):
                outs.append(max_depth_from(no, depth + 1))
            return max(outs) if outs else depth

        per_tree_depth.append(max_depth_from(root, 0))

        sf = g.loc[g[feat_col] != "Leaf", feat_col].tolist()
        split_features.extend(sf)

    counts = pd.Series(split_features).value_counts()
    if len(counts) == 0:
        split_entropy_bits = 0.0
    else:
        split_entropy_bits = float(scipy_entropy(counts, base=2))

    return {
        "n_trees": n_trees,
        "total_leaves": int(sum(per_tree_leaves)),
        "mean_leaves_per_tree": float(np.mean(per_tree_leaves)),
        "std_leaves_per_tree": float(np.std(per_tree_leaves, ddof=0)),
        "max_leaves_single_tree": int(np.max(per_tree_leaves)),
        "mean_max_depth_per_tree": float(np.mean(per_tree_depth)),
        "max_depth_any_tree": int(np.max(per_tree_depth)),
        "n_distinct_split_features": int(len(set(split_features))),
        "split_feature_entropy_bits": split_entropy_bits,
    }


CSV_FIELDS = [
    "dataset",
    "fold",
    "trial_id",
    "base_seed",
    "random_state",
    "n_estimators",
    "max_depth",
    "n_features",
    "n_train_samples",
    "fit_time_sec",
    "train_acc",
    "val_acc",
    "test_acc",
    "n_trees",
    "total_leaves",
    "mean_leaves_per_tree",
    "std_leaves_per_tree",
    "max_leaves_single_tree",
    "mean_max_depth_per_tree",
    "max_depth_any_tree",
    "n_distinct_split_features",
    "split_feature_entropy_bits",
]


def append_csv_row(path: str, row: Dict[str, Any]) -> None:
    """
    Append one row and flush + fsync so partial runs (crash/kill) keep all rows
    written so far on disk.
    """
    path = os.path.abspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)
        f.flush()
        try:
            os.fsync(f.fileno())
        except OSError:
            pass


def run_experiment(args: argparse.Namespace) -> None:
    home_dir = os.path.abspath(os.path.expanduser(args.home_dir))

    out_path = args.output_csv
    if not out_path:
        out_path = os.path.join(
            home_dir, "processed_capability_results", "xgb_capability_comparison.csv"
        )
    else:
        out_path = os.path.abspath(os.path.expanduser(out_path))

    if args.overwrite_csv and os.path.isfile(out_path):
        os.remove(out_path)
        print(f"Removed existing CSV (--overwrite-csv): {out_path}")

    rows_before = 0
    if os.path.isfile(out_path):
        try:
            rows_before = len(pd.read_csv(out_path))
        except (OSError, ValueError, pd.errors.ParserError):
            rows_before = 0

    trial_id = args.trial_id
    base_seed = args.base_seed
    # Match shapefbt_capability_run.py: one random_state per SLURM-style trial_id
    random_state = base_seed + trial_id

    total_runs = (
        len(args.datasets)
        * len(args.folds)
        * len(args.n_estimators)
        * len(args.max_depth)
    )
    done = 0

    for dataset in args.datasets:
        data_factory = DataFactory_clf(
            dataset, cache_dir=os.path.join(home_dir, "data")
        )
        for fold in args.folds:
            X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(
                fold
            )
            n_features = int(X_train.shape[1])
            n_train = int(X_train.shape[0])

            for n_est in args.n_estimators:
                for max_dep in args.max_depth:
                    done += 1
                    print(
                        f"[{done}/{total_runs}] {dataset} fold={fold} "
                        f"n_estimators={n_est} max_depth={max_dep}",
                        flush=True,
                    )

                    clf = _make_xgb(
                        n_estimators=n_est,
                        max_depth=max_dep,
                        random_state=random_state,
                        n_jobs=args.n_jobs,
                        device=args.device,
                    )

                    t0 = time.perf_counter()
                    clf.fit(X_train, y_train)
                    fit_time = time.perf_counter() - t0

                    y_tr_p = clf.predict(X_train)
                    y_va_p = clf.predict(X_val)
                    y_te_p = clf.predict(X_test)
                    train_acc = float(accuracy_score(y_train, y_tr_p))
                    val_acc = float(accuracy_score(y_val, y_va_p))
                    test_acc = float(accuracy_score(y_test, y_te_p))

                    struct = forest_structure_metrics(clf)

                    row: Dict[str, Any] = {
                        "dataset": dataset,
                        "fold": fold,
                        "trial_id": trial_id,
                        "base_seed": base_seed,
                        "random_state": random_state,
                        "n_estimators": n_est,
                        "max_depth": max_dep,
                        "n_features": n_features,
                        "n_train_samples": n_train,
                        "fit_time_sec": fit_time,
                        "train_acc": train_acc,
                        "val_acc": val_acc,
                        "test_acc": test_acc,
                        **struct,
                    }

                    append_csv_row(out_path, row)

    # Sanity check: one row per (dataset, fold, n_estimators, max_depth) for this run
    try:
        check = pd.read_csv(out_path)
        actual_total = len(check)
        added = actual_total - rows_before
        print(f"CSV path: {out_path}")
        print(
            f"Rows in file: {actual_total} "
            f"(+{added} this run; expected +{total_runs} this run)"
        )
        if added < total_runs:
            print(
                "Warning: fewer new rows than expected — run interrupted or an error "
                "occurred mid-loop. Use --overwrite-csv and re-run for a full clean grid."
            )
        dup = check.duplicated(
            subset=["dataset", "fold", "n_estimators", "max_depth"], keep=False
        )
        if dup.any():
            print(
                f"Warning: {dup.sum()} rows participate in duplicate "
                "(dataset, fold, n_estimators, max_depth) keys — use --overwrite-csv "
                "for a single clean run."
            )
    except Exception as e:
        print(f"Finished run (could not verify CSV: {e})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="XGBoost forest structure comparison across capability grid (CSV log)."
    )
    parser.add_argument(
        "--home-dir",
        type=str,
        default=ROOT,
        help="Project root (data cache, output paths). Default: directory of this script.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=["magic", "raisin", "bidding"],
        help="DataFactory_clf dataset names.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3, 4],
        help="Fold indices (same as DataFactory_clf.get_data).",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        nargs="+",
        default=[10, 20, 40, 60, 80, 100],
        help="Grid of n_estimators (same spirit as capability experiments).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        nargs="+",
        default=[3, 4, 5],
        help="Grid of max_depth.",
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        default=0,
        help="Added to base-seed for random_state (matches shapefbt_capability_run).",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=42,
        help="Base seed; random_state = base_seed + trial_id.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="XGBoost n_jobs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="XGBoost device (e.g. cpu, cuda).",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="",
        help="Output CSV path. Default: <home-dir>/processed_capability_results/xgb_capability_comparison.csv",
    )
    parser.add_argument(
        "--overwrite-csv",
        action="store_true",
        help="Delete existing output CSV before running (recommended for a full clean grid).",
    )

    args = parser.parse_args()
    args.output_csv = args.output_csv.strip()
    args.home_dir = os.path.abspath(os.path.expanduser(args.home_dir))
    run_experiment(args)


if __name__ == "__main__":
    main()
