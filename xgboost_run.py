"""
Usage:
  python xgboost_run.py --dataset room --trial-id 0 --folds 0 1 2 3 4 --max-depth-choices 1 2 3 4 -1 --n-trials 20
  Use --max-depth-choices -1 to allow max_depth=None in the search space.
  Writes one JSON results file per fold under results/<dataset>/ (best trial per fold).
  Hyperparameters tuned:
  - max_depth: 1..4, or None (unconstrained)
  - gamma: 0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
  - n_estimators: 50, 100, 200, 400, 800
  - min_child_weight: 1.0, 2.0, 5.0, 10.0
"""

import os
import json
import time
import argparse

import numpy as np

from sklearn.metrics import accuracy_score

import xgboost as xgb


def sample_hyperparameters(rng: np.random.Generator, max_depth_choices: list[int]):
    """
    Sample XGBoost hyperparameters intended to be comparable to ShapeFBT knobs.

    Mapping intuition
    ---------------
    - depth (outer_tree_max_depth)            -> max_depth (tuned here)
    - min_impurity_decrease (split threshold) -> gamma (min loss reduction to split)
    - max_number_of_conjunctions (complexity/runtime) -> n_estimators
    - min_child_weight -> QtoNakul: can we compare this to min_conjunctions_split? (not the same thing, btu tried to have a similar effect and not split weak/small children)
    """
    depth_choice = int(rng.choice(max_depth_choices))
    max_depth = None if depth_choice == -1 else depth_choice
    gamma = float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]))
    n_estimators = int(rng.choice([50, 100, 200, 400, 800]))

    # A mild additional regularizer that plays a similar role to
    # "minimum support" constraints in tree growth.
    min_child_weight = float(rng.choice([1.0, 2.0, 5.0, 10.0]))

    return {
        "max_depth": max_depth,
        "gamma": gamma,
        "n_estimators": n_estimators,
        "min_child_weight": min_child_weight,
    }


def prepare_and_save_results(results: dict, args: argparse.Namespace, fold_idx: int):
    # Convert numeric results to strings for stable JSON formatting
    serialized_results = {}
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            serialized_results[key] = f"{float(value):.6f}"
        else:
            serialized_results[key] = value

    output = {
        "model": "XGBoost",
        "trial_id": str(args.trial_id),
        "fold": str(fold_idx),
        "dataset": args.dataset,
        "random_seed": str(args.random_seed),
        "base_seed": str(args.base_seed),
        "n_trials": str(args.n_trials),
    }
    output.update(serialized_results)

    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(
        output_path,
        f"xgboost_results_trial_{args.dataset}_{args.trial_id}_{fold_idx}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost hyperparameter tuner")
    parser.add_argument("--dataset", type=str, default="room", help="Dataset name. Default is room")
    parser.add_argument("--trial-id", type=int, required=True, help="Trial ID (typically from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--max-depth-choices",
        type=int,
        nargs="+",
        default=[2, 3, 4, -1],
        help="Search space for max_depth. Include -1 to represent None.",
    )
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="List of fold indices to run. Default [0].",
    )
    parser.add_argument("--timeout", type=int, default=60 * 60, help="Timeout (seconds) for the whole run")
    parser.add_argument("--home-dir", type=str, default="./", help="Home/output directory")
    parser.add_argument("--min_time", type=int, default=3, help="Minimum wall time (seconds) before exit")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of hyperparameter trials at this depth")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.home_dir, "results")

    t0 = time.time()

    # Seed per trial_id (same pattern as shapecart_run.py / shapefbt_run.py)
    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)

    # Import data factory
    try:
        from data_utils import DataFactory_clf  # type: ignore
    except Exception:
        from importlib.machinery import SourceFileLoader

        data_utils_path = os.path.expanduser("~/Downloads/data_utils (1).py")
        data_utils_mod = SourceFileLoader("data_utils_download", data_utils_path).load_module()
        DataFactory_clf = data_utils_mod.DataFactory_clf

    data_factory = DataFactory_clf(args.dataset, cache_dir=os.path.join(args.home_dir, "data"))

    run_start = time.time()
    for fold_idx in args.folds:
        if time.time() - run_start > args.timeout:
            break

        X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(fold_idx=fold_idx)

        best = {
            "val_acc": -np.inf,
            "params": None,
            "train_acc": None,
            "test_acc": None,
            "elapsed_time": None,
        }

        start_fit = time.time()
        for _ in range(args.n_trials):
            if time.time() - run_start > args.timeout:
                break

            hp = sample_hyperparameters(rng, max_depth_choices=args.max_depth_choices)

            clf = xgb.XGBClassifier(
                random_state=args.random_seed,
                max_depth=hp["max_depth"],
                gamma=hp["gamma"],
                n_estimators=hp["n_estimators"],
                min_child_weight=hp["min_child_weight"],
                eval_metric="logloss",
            )

            t1 = time.time()
            clf.fit(X_train, y_train)
            t2 = time.time()

            train_pred = clf.predict(X_train)
            val_pred = clf.predict(X_val)
            test_pred = clf.predict(X_test)

            train_acc = accuracy_score(y_train, train_pred)
            val_acc = accuracy_score(y_val, val_pred)
            test_acc = accuracy_score(y_test, test_pred)

            if val_acc > best["val_acc"]:
                best = {
                    "val_acc": val_acc,
                    "params": hp,
                    "train_acc": train_acc,
                    "test_acc": test_acc,
                    "elapsed_time": t2 - t1,
                }

        results = {
            "train_acc": best["train_acc"] if best["train_acc"] is not None else float("nan"),
            "val_acc": best["val_acc"] if best["val_acc"] != -np.inf else float("nan"),
            "test_acc": best["test_acc"] if best["test_acc"] is not None else float("nan"),
            "elapsed_time": best["elapsed_time"] if best["elapsed_time"] is not None else float("nan"),
            "best_params": best["params"],
            "fold_fit_time": time.time() - start_fit,
        }

        prepare_and_save_results(results, args, fold_idx=fold_idx)

    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)

