"""
Usage:
  # Default backend is DataFactory (ShapeCART-style)
  python sagifbt_run.py --dataset room --trial-id 0 --folds 0 1 2 3 4 --depths 2 3 4 --n-trials 20
  python sagifbt_run.py --dataset room --trial-id 0 --data-backend datafactory --datafactory-module src.data_utils --datafactory-class DataFactory_clf

  # Opt in to legacy datasets.py backend
  python sagifbt_run.py --dataset iris --trial-id 0 --data-backend datasets --folds 0 1 2 3 4 --depths 1 2 3 4 --n-trials 20
  python sagifbt_run.py --datasets iris breast_cancer --trial-id 0 --data-backend datasets --folds 0 1 2 3 4 --depths 1 2 3 4 --n-trials 20
  python sagifbt_run.py --all-datasets --trial-id 0 --data-backend datasets --folds 0 --depths 2 --n-trials 5

What it does
------------
- Runs per-fold, per-depth randomized hyperparameter search where each trial samples both
  XGBoost parameters (the source model) and FBT parameters (the approximation model), then keeps
  the pair with the best validation accuracy.
- Supports two data loading modes: DataFactory (`get_data(fold)`, default) and legacy
  `datasets.py` deterministic fold splitting (opt-in with `--data-backend datasets`).
- Writes one JSON result file per (fold, depth) under `results/<dataset>/`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

import xgboost as xgb

from FBT import FBT
from datasets import datasets_dict


@dataclass(frozen=True)
class Split:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    feature_cols: list[str]
    label_col: str


def _stable_jsonify(results: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in results.items():
        if isinstance(v, (float, np.floating)):
            out[k] = f"{float(v):.6f}"
        else:
            out[k] = v
    return out


def _load_full_dataset(dataset: str, random_state: int) -> tuple[pd.DataFrame, list[str], str]:
    """
    `datasets.py` getters return (train, test, feature_cols, label_col) based on a random split.
    For fold-based experiments we combine train+test and do our own CV splits deterministically.
    """
    getter = datasets_dict.get(dataset)
    if getter is None:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Available: {sorted(datasets_dict.keys())}"
        )
    train, test, feature_cols, label_col = getter(random_state=random_state)
    full = pd.concat([train, test], axis=0, ignore_index=True)
    return full, list(feature_cols), str(label_col)


def _to_dataframe_split(
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    X_test: Any,
    y_test: Any,
) -> Split:
    """
    Convert array-like splits returned by a DataFactory into the DataFrame-based
    Split format expected by the FBT training/eval pipeline.
    """
    X_train_np = np.asarray(X_train)
    X_val_np = np.asarray(X_val)
    X_test_np = np.asarray(X_test)
    y_train_np = np.asarray(y_train)
    y_val_np = np.asarray(y_val)
    y_test_np = np.asarray(y_test)

    if X_train_np.ndim != 2:
        raise ValueError(f"Expected X_train to be 2D, got shape={X_train_np.shape}")
    n_features = int(X_train_np.shape[1])
    feature_cols = [f"f{i}" for i in range(n_features)]
    label_col = "label"

    train_df = pd.DataFrame(X_train_np, columns=feature_cols)
    train_df[label_col] = y_train_np

    val_df = pd.DataFrame(X_val_np, columns=feature_cols)
    val_df[label_col] = y_val_np

    test_df = pd.DataFrame(X_test_np, columns=feature_cols)
    test_df[label_col] = y_test_np

    return Split(
        train=train_df,
        val=val_df,
        test=test_df,
        feature_cols=feature_cols,
        label_col=label_col,
    )


def _make_datafactory(
    *,
    dataset: str,
    module_path: str,
    class_name: str,
    cache_dir: str,
) -> Any:
    """
    Instantiate a data factory compatible with ShapeCART script expectations:
    factory = DataFactory_clf(dataset, cache_dir=...)
    """
    mod = importlib.import_module(module_path)
    factory_cls = getattr(mod, class_name, None)
    if factory_cls is None:
        raise AttributeError(f"Could not find class '{class_name}' in module '{module_path}'")
    return factory_cls(dataset, cache_dir=cache_dir)


def make_fold_split(
    full: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    fold_idx: int,
    *,
    n_folds: int,
    base_seed: int,
    val_ratio_within_train: float = 0.2,
) -> Split:
    y = full[label_col].values
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=base_seed)
    folds = list(skf.split(np.zeros(len(full)), y))
    if fold_idx < 0 or fold_idx >= len(folds):
        raise ValueError(f"fold_idx={fold_idx} out of range for n_folds={n_folds}")

    train_val_idx, test_idx = folds[fold_idx]
    train_val = full.iloc[train_val_idx].reset_index(drop=True)
    test = full.iloc[test_idx].reset_index(drop=True)

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_ratio_within_train,
        random_state=base_seed + fold_idx,
    )
    y_tv = train_val[label_col].values
    (train_idx, val_idx) = next(sss.split(np.zeros(len(train_val)), y_tv))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    return Split(train=train, val=val, test=test, feature_cols=feature_cols, label_col=label_col)


def sample_fbt_hyperparameters(rng: np.random.Generator) -> dict[str, Any]:
    """
    Sample hyperparameters that exist in `FBT.__init__`.
    """
    min_forest_size = int(rng.choice([5, 10, 20, 30, 50]))
    max_number_of_conjunctions = int(rng.choice([200, 500, 1000, 2000, 5000]))
    pruning_method = rng.choice([None, "auc"])
    return {
        "min_forest_size": min_forest_size,
        "max_number_of_conjunctions": max_number_of_conjunctions,
        "pruning_method": pruning_method,
    }


def sample_xgb_hyperparameters(rng: np.random.Generator, *, num_class: int) -> dict[str, Any]:
    """
    Sample XGBoost hyperparameters for the *source model* passed into FBT.

    Important: FBT extracts conjunctions from `model._Booster.get_dump()`, so we mostly
    tune params that change tree structure/size.
    """
    max_depth = int(rng.choice([2, 3, 4, 5, 7, 10]))
    n_estimators = int(rng.choice([50, 100, 200, 400, 800]))
    learning_rate = float(rng.choice([0.01, 0.05, 0.1, 0.2]))
    subsample = float(rng.choice([0.6, 0.8, 1.0]))
    colsample_bytree = float(rng.choice([0.6, 0.8, 1.0]))
    min_child_weight = float(rng.choice([1.0, 2.0, 5.0, 10.0]))
    gamma = float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]))
    reg_alpha = float(rng.choice([0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]))
    reg_lambda = float(rng.choice([0.1, 1.0, 10.0, 100.0]))

    objective = "multi:softprob" if num_class > 2 else "binary:logitraw"

    return {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "subsample": subsample,
        "colsample_bytree": colsample_bytree,
        "min_child_weight": min_child_weight,
        "gamma": gamma,
        "reg_alpha": reg_alpha,
        "reg_lambda": reg_lambda,
        "objective": objective,
    }


def prepare_and_save_results(
    results: dict[str, Any],
    args: argparse.Namespace,
    *,
    depth: int,
    fold_idx: int,
) -> None:
    output = {
        "model": "SagiFBT",
        "trial_id": str(args.trial_id),
        "fold": str(fold_idx),
        "dataset": args.dataset,
        "random_seed": str(args.random_seed),
        "base_seed": str(args.base_seed),
        "max_depth": str(depth),
        "n_trials": str(args.n_trials),
        "n_folds": str(args.n_folds),
    }
    output.update(_stable_jsonify(results))

    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(
        output_path,
        f"sagifbt_results_trial_{depth}_{args.dataset}_{args.trial_id}_{fold_idx}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def _fit_xgb_classifier(
    split: Split,
    *,
    rng: np.random.Generator,
    random_seed: int,
) -> tuple[xgb.XGBClassifier, dict[str, Any]]:
    num_class = int(split.train[split.label_col].nunique())
    hp = sample_xgb_hyperparameters(rng, num_class=num_class)

    clf = xgb.XGBClassifier(
        random_state=random_seed,
        max_depth=hp["max_depth"],
        n_estimators=hp["n_estimators"],
        learning_rate=hp["learning_rate"],
        subsample=hp["subsample"],
        colsample_bytree=hp["colsample_bytree"],
        min_child_weight=hp["min_child_weight"],
        gamma=hp["gamma"],
        reg_alpha=hp["reg_alpha"],
        reg_lambda=hp["reg_lambda"],
        objective=hp["objective"],
        eval_metric="logloss",
        n_jobs=1,
    )

    X_train = split.train[split.feature_cols].values
    y_train = split.train[split.label_col].values
    clf.fit(X_train, y_train)
    return clf, hp


def _eval_fbt(
    model: FBT,
    split: Split,
) -> tuple[float, float, float]:
    X_train = split.train[split.feature_cols]
    y_train = split.train[split.label_col].values
    X_val = split.val[split.feature_cols]
    y_val = split.val[split.label_col].values
    X_test = split.test[split.feature_cols]
    y_test = split.test[split.label_col].values

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)

    return (
        float(accuracy_score(y_train, train_pred)),
        float(accuracy_score(y_val, val_pred)),
        float(accuracy_score(y_test, test_pred)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sagi FBT hyperparameter tuner per depth")
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Single dataset name (see datasets.py). Ignored if --datasets/--all-datasets is used.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        default=None,
        help="One or more dataset names (see datasets.py).",
    )
    parser.add_argument(
        "--all-datasets",
        action="store_true",
        help="Run over all datasets in datasets.py (datasets_dict keys).",
    )
    parser.add_argument("--trial-id", type=int, required=True, help="Trial ID (typically from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2, 3, 4],
        help="List of FBT max_depth values to run.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="List of fold indices to run. Default [0].",
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Total number of CV folds. Default 5.")
    parser.add_argument("--timeout", type=int, default=60 * 60, help="Timeout (seconds) for the whole run")
    parser.add_argument("--home-dir", type=str, default="./", help="Home/output directory")
    parser.add_argument("--min_time", type=int, default=3, help="Minimum wall time (seconds) before exit")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of hyperparameter trials at this depth")
    parser.add_argument(
        "--data-backend",
        type=str,
        default="datafactory",
        choices=["datasets", "datafactory"],
        help="Data loading backend: 'datafactory' (default, ShapeCART-style) or 'datasets' (legacy datasets.py).",
    )
    parser.add_argument(
        "--datafactory-module",
        type=str,
        default="src.data_utils",
        help="Python module that contains the DataFactory class (used when --data-backend datafactory).",
    )
    parser.add_argument(
        "--datafactory-class",
        type=str,
        default="DataFactory_clf",
        help="Class name of the DataFactory to instantiate (used when --data-backend datafactory).",
    )
    parser.add_argument(
        "--data-cache-dir",
        type=str,
        default=None,
        help="Cache dir passed to DataFactory. Default: <home-dir>/data",
    )
    args = parser.parse_args()

    args.output_dir = os.path.join(args.home_dir, "results")

    t0 = time.time()

    # Seed per trial_id (same pattern as other run scripts)
    trial_seed = int(args.base_seed + args.trial_id)
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)

    if args.all_datasets:
        dataset_list = sorted(datasets_dict.keys())
    elif args.datasets is not None and len(args.datasets) > 0:
        dataset_list = list(args.datasets)
    else:
        dataset_list = [args.dataset or "iris"]

    run_start = time.time()
    for dataset_name in dataset_list:
        if time.time() - run_start > args.timeout:
            break

        # Make sure output JSON says the right dataset (even though args is shared).
        args.dataset = dataset_name

        if args.data_backend == "datasets":
            # Fold splits are deterministic from base_seed; we still load per dataset.
            full, feature_cols, label_col = _load_full_dataset(dataset_name, random_state=args.random_seed)
            data_factory = None
        else:
            cache_dir = args.data_cache_dir or os.path.join(args.home_dir, "data")
            data_factory = _make_datafactory(
                dataset=dataset_name,
                module_path=args.datafactory_module,
                class_name=args.datafactory_class,
                cache_dir=cache_dir,
            )
            full, feature_cols, label_col = None, None, None

        for fold_idx in args.folds:
            if time.time() - run_start > args.timeout:
                break

            if args.data_backend == "datasets":
                split = make_fold_split(
                    full,
                    feature_cols=feature_cols,
                    label_col=label_col,
                    fold_idx=fold_idx,
                    n_folds=int(args.n_folds),
                    base_seed=int(args.base_seed),
                )
            else:
                # ShapeCART-style DataFactory interface:
                # X_train, y_train, X_val, y_val, X_test, y_test = factory.get_data(fold_idx)
                X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(int(fold_idx))
                split = _to_dataframe_split(
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                )

            for depth in args.depths:
                if time.time() - run_start > args.timeout:
                    break

                best = {
                    "val_acc": -np.inf,
                    "train_acc": None,
                    "test_acc": None,
                    "fbt_params": None,
                    "xgb_params": None,
                    "xgb_fit_time": None,
                    "fbt_fit_time": None,
                    "trial_elapsed_time": None,
                }

                depth_start = time.time()
                for _ in range(int(args.n_trials)):
                    if time.time() - run_start > args.timeout:
                        break

                    fbt_hp = sample_fbt_hyperparameters(rng)

                    # Train a source XGBoost (this is what FBT approximates)
                    xgb_t1 = time.time()
                    xgb_model, xgb_hp = _fit_xgb_classifier(
                        split,
                        rng=rng,
                        random_seed=int(args.random_seed),
                    )
                    xgb_t2 = time.time()

                    # Train FBT approximation on training split
                    model = FBT(
                        max_depth=int(depth),
                        min_forest_size=int(fbt_hp["min_forest_size"]),
                        max_number_of_conjunctions=int(fbt_hp["max_number_of_conjunctions"]),
                        pruning_method=fbt_hp["pruning_method"],
                    )

                    fbt_t1 = time.time()
                    model.fit(
                        split.train,
                        feature_cols=split.feature_cols,
                        label_col=split.label_col,
                        xgb_model=xgb_model,
                    )
                    fbt_t2 = time.time()

                    train_acc, val_acc, test_acc = _eval_fbt(model, split)

                    if val_acc > best["val_acc"]:
                        best = {
                            "val_acc": val_acc,
                            "train_acc": train_acc,
                            "test_acc": test_acc,
                            "fbt_params": fbt_hp,
                            "xgb_params": xgb_hp,
                            "xgb_fit_time": xgb_t2 - xgb_t1,
                            "fbt_fit_time": fbt_t2 - fbt_t1,
                            "trial_elapsed_time": (xgb_t2 - xgb_t1) + (fbt_t2 - fbt_t1),
                        }

                results = {
                    "train_acc": best["train_acc"] if best["train_acc"] is not None else float("nan"),
                    "val_acc": best["val_acc"] if best["val_acc"] != -np.inf else float("nan"),
                    "test_acc": best["test_acc"] if best["test_acc"] is not None else float("nan"),
                    "best_fbt_params": best["fbt_params"],
                    "best_xgb_params": best["xgb_params"],
                    "best_xgb_fit_time": best["xgb_fit_time"] if best["xgb_fit_time"] is not None else float("nan"),
                    "best_fbt_fit_time": best["fbt_fit_time"] if best["fbt_fit_time"] is not None else float("nan"),
                    "best_trial_elapsed_time": best["trial_elapsed_time"]
                    if best["trial_elapsed_time"] is not None
                    else float("nan"),
                    "depth_fit_time": time.time() - depth_start,
                }

                prepare_and_save_results(results, args, depth=int(depth), fold_idx=int(fold_idx))

    final_elapsed = time.time() - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)

