"""
Usage:
  python shapefbt_run.py --dataset room --trial-id 0 --folds 0 1 2 3 4 --depths 1 2 3 4 --n-trials 20
  ^loops through all folds and depths. can also run with --folds 0 to run only on fold 0.
  Use --depths -1 to run with outer_tree_max_depth=None for that run.
  Writes one JSON results file per (fold, depth) under results/<dataset>/.
  Current hyperparameters tuned (for each outer tree depth):
  - inner_tree_max_depth: 1..8
  - inner_tree_criterion: "entropy" or "gini"
  - max_number_of_conjunctions: 200, 500, 1000, 2000, 5000
  - min_impurity_decrease: 0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
  - min_conjunctions_split: 2, 5, 10, 20
  - xgb max_depth: 2, 3, 4, 5, 7, 10
  - xgb n_estimators: 50, 100, 200, 400, 800
  - xgb learning_rate: 0.01, 0.05, 0.1, 0.2
  - xgb subsample: 0.6, 0.8, 1.0
  - xgb colsample_bytree: 0.6, 0.8, 1.0
  - xgb min_child_weight: 1.0, 2.0, 5.0, 10.0
  - xgb gamma: 0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1
  - xgb reg_alpha: 0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0
  - xgb reg_lambda: 0.1, 1.0, 10.0, 100.0
"""

import os
import json
import time
import argparse

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score

import xgboost as xgb

from ShapeFBT import ShapeFBT


def sample_hyperparameters(rng: np.random.Generator):
    """
    Randomly sample ShapeFBT/FBT-related hyperparameters for tuning.
    """
    # Outer/inner tree relationship:
    # - outer_tree_max_depth is fixed by --depths (per-depth loop)
    # - inner_tree_max_depth is tuned as a small integer (or tied to outer)
    inner_tree_max_depth = int(rng.integers(1, 9))  # 1..8

    # Inner-tree impurity criterion (used inside FBT bucketing)
    inner_tree_criterion = str(rng.choice(["entropy", "gini"]))

    # Pruning: keep same option that your current code uses in test_script.py
    pruning_method = "auc"

    # Conjunction-set / pruning controls
    # min_forest_size = int(rng.choice([5, 10, 20, 30, 50]))
    min_forest_size = 10
    max_number_of_conjunctions = int(rng.choice([200, 500, 1000, 2000, 5000]))

    # Outer-tree split acceptance threshold (ShapeFBT checks impurity decrease from map_to_buckets)
    min_impurity_decrease = float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]))

    # Minimum number of conjunctions required at a node to split (outer tree)
    min_conjunctions_split = int(rng.choice([2, 5, 10, 20]))

    # Coordinate-descent bucketing parameter (currently used as k in map_to_buckets)
    k = int(rng.choice([2]))

    return {
        "inner_tree_max_depth": inner_tree_max_depth,
        "inner_tree_criterion": inner_tree_criterion,
        "pruning_method": pruning_method,
        "min_forest_size": min_forest_size,
        "max_number_of_conjunctions": max_number_of_conjunctions,
        "min_impurity_decrease": min_impurity_decrease,
        "min_conjunctions_split": min_conjunctions_split,
        "k": k,
    }


def sample_xgb_hyperparameters(rng: np.random.Generator, *, num_class: int):
    """
    Sample XGBoost hyperparameters for the source model ShapeFBT approximates.
    Search space intentionally matches sagifbt_run.py.
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


def fit_xgb_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    rng: np.random.Generator,
    random_seed: int,
):
    num_class = int(np.unique(y_train).shape[0])
    hp = sample_xgb_hyperparameters(rng, num_class=num_class)
    xgb_model = xgb.XGBClassifier(
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
    xgb_model.fit(X_train, y_train)
    return xgb_model, hp


def prepare_and_save_results(results: dict, args: argparse.Namespace, depth: int, fold_idx: int):
    # Convert numeric results to strings for stable JSON formatting
    serialized_results = {}
    for key, value in results.items():
        if isinstance(value, (float, np.floating)):
            serialized_results[key] = f"{float(value):.6f}"
        else:
            serialized_results[key] = value

    output = {
        "model": "ShapeFBT",
        "trial_id": str(args.trial_id),
        "fold": str(fold_idx),
        "dataset": args.dataset,
        "random_seed": str(args.random_seed),
        "base_seed": str(args.base_seed),
        "outer_tree_max_depth": str(depth) if depth != -1 else "",
        "n_trials": str(args.n_trials),
    }
    output.update(serialized_results)

    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(
        output_path,
        f"shapefbt_results_trial_{depth}_{args.dataset}_{args.trial_id}_{fold_idx}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


def build_train_df(X: np.ndarray, y: np.ndarray, label_col: str = "y"):
    """
    Wrap numpy arrays into a DataFrame compatible with ShapeFBT.fit.
    """
    n_features = X.shape[1]
    feature_cols = [f"f{i}" for i in range(n_features)]
    train_df = pd.DataFrame(X, columns=feature_cols)
    train_df[label_col] = y
    return train_df, feature_cols, label_col


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ShapeFBT hyperparameter tuner per outer depth")
    parser.add_argument("--dataset", type=str, default="room", help="Dataset name. Default is room")
    parser.add_argument("--trial-id", type=int, required=True, help="Trial ID (typically from SLURM_ARRAY_TASK_ID)")
    parser.add_argument("--base-seed", type=int, default=42, help="Base random seed")
    parser.add_argument(
        "--depths",
        type=int,
        nargs="+",
        default=[2],
        help="List of outer tree depths to run. Use -1 to represent None.",
    )
    parser.add_argument("--timeout", type=int, default=60 * 60, help="Timeout (seconds) for the whole run")
    parser.add_argument("--home-dir", type=str, default="./", help="Home/output directory")
    parser.add_argument(
        "--folds",
        type=int,
        nargs="+",
        default=[0],
        help="List of fold indices to run. Default [0].",
    )
    parser.add_argument("--min_time", type=int, default=3, help="Minimum wall time (seconds) before exit")
    parser.add_argument("--n-trials", type=int, default=20, help="Number of hyperparameter trials at this depth")
    args = parser.parse_args()

    args.output_dir = os.path.join(args.home_dir, "results")

    t0 = time.time()

    # Seed per trial_id (same pattern as shapecart_run.py)
    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)

    # Import data factory (expected to be available to the run environment)
    # `data_utils (1).py` is a user download; in practice you should point to the actual module path you use.
    try:
        from data_utils import DataFactory_clf  # type: ignore
    except Exception:
        # fallback for the downloaded filename in your workspace
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

        # Prepare DataFrames for ShapeFBT (it expects DF + column names)
        train_df, feature_cols, label_col = build_train_df(X_train, y_train, label_col="y")
        val_df, _, _ = build_train_df(X_val, y_val, label_col="y")
        test_df, _, _ = build_train_df(X_test, y_test, label_col="y")

        for depth in args.depths:
            if time.time() - run_start > args.timeout:
                break

            # Depth handling per run
            if depth == -1:
                outer_tree_max_depth = None
            else:
                outer_tree_max_depth = int(depth)

            # Tune hyperparameters at this specific outer depth using validation accuracy
            best = {
                "val_acc": -np.inf,
                "params": None,
                "xgb_params": None,
                "train_acc": None,
                "test_acc": None,
                "xgb_fit_time": None,
                "shapefbt_fit_time": None,
                "trial_elapsed_time": None,
            }

            start_fit = time.time()
            for _ in range(args.n_trials):
                if time.time() - run_start > args.timeout:
                    break

                hp = sample_hyperparameters(rng)
                xgb_t1 = time.time()
                xgb_model, xgb_hp = fit_xgb_classifier(
                    X_train,
                    y_train,
                    rng=rng,
                    random_seed=int(args.random_seed),
                )
                xgb_t2 = time.time()

                model = ShapeFBT(
                    outer_tree_max_depth=outer_tree_max_depth,
                    inner_tree_max_depth=hp["inner_tree_max_depth"],
                    inner_tree_criterion=hp["inner_tree_criterion"],
                    min_forest_size=hp["min_forest_size"],
                    max_number_of_conjunctions=hp["max_number_of_conjunctions"],
                    pruning_method=hp["pruning_method"],
                    min_conjunctions_split=hp["min_conjunctions_split"],
                    min_impurity_decrease=hp["min_impurity_decrease"],
                    k=hp["k"],
                    verbose=False,
                )

                t1 = time.time()
                model.fit(
                    train_df,
                    feature_cols=feature_cols,
                    label_col=label_col,
                    xgb_model=xgb_model,
                    feature_dict=getattr(data_factory, "feature_dict", None),
                )
                t2 = time.time()

                # Evaluate
                train_pred, _ = model.predict_Xy(train_df[feature_cols])
                val_pred, _ = model.predict_Xy(val_df[feature_cols])
                test_pred, _ = model.predict_Xy(test_df[feature_cols])

                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                test_acc = accuracy_score(y_test, test_pred)

                if val_acc > best["val_acc"]:
                    best = {
                        "val_acc": val_acc,
                        "params": hp,
                        "xgb_params": xgb_hp,
                        "train_acc": train_acc,
                        "test_acc": test_acc,
                        "xgb_fit_time": xgb_t2 - xgb_t1,
                        "shapefbt_fit_time": t2 - t1,
                        "trial_elapsed_time": (xgb_t2 - xgb_t1) + (t2 - t1),
                    }

            results = {
                "train_acc": best["train_acc"] if best["train_acc"] is not None else float("nan"),
                "val_acc": best["val_acc"] if best["val_acc"] != -np.inf else float("nan"),
                "test_acc": best["test_acc"] if best["test_acc"] is not None else float("nan"),
                "elapsed_time": best["shapefbt_fit_time"] if best["shapefbt_fit_time"] is not None else float("nan"),
                "best_params": best["params"],
                "best_shapefbt_params": best["params"],
                "best_xgb_params": best["xgb_params"],
                "best_xgb_fit_time": best["xgb_fit_time"] if best["xgb_fit_time"] is not None else float("nan"),
                "best_shapefbt_fit_time": best["shapefbt_fit_time"]
                if best["shapefbt_fit_time"] is not None
                else float("nan"),
                "best_trial_elapsed_time": best["trial_elapsed_time"]
                if best["trial_elapsed_time"] is not None
                else float("nan"),
                "depth_fit_time": time.time() - start_fit,
            }

            prepare_and_save_results(results, args, depth=depth, fold_idx=fold_idx)

    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)
