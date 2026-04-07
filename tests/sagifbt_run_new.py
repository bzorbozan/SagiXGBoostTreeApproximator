import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_utils import *
from xgboost import XGBClassifier
from FBT import FBT
from sklearn.metrics import accuracy_score
import signal
import time
import json
import argparse
import numpy as np
import pandas as pd

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class TimeoutException(Exception):
    pass


def handler(signum, frame):
    raise TimeoutException()


def sample_hyperparameters(rng):
    """
    Sample a set of FBT hyperparameters.

    FBT also requires a pre-trained XGBoost model, so XGBoost
    hyperparameters are sampled here too.

    Returns:
        dict: Sampled hyperparameters
    """
    # ── XGBoost base model ────────────────────────────────────────────────
    xgb_max_depth = int(rng.integers(3, 9))
    xgb_n_estimators = int(rng.choice([50, 100, 250, 500, 750, 1000]))
    xgb_learning_rate = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
    xgb_colsample_bytree = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
    xgb_subsample = float(rng.choice([0.5, 0.63, 0.8, 1.0]))
    xgb_min_child_weight = int(rng.choice([1, 3, 5, 10]))
    xgb_reg_lambda = float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0]))
    xgb_reg_alpha = float(rng.choice([0.0, 0.1, 0.5, 1.0]))
    xgb_gamma = float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))

    # ── FBT ───────────────────────────────────────────────────────────────
    max_depth = int(rng.integers(2, 7))
    max_number_of_conjunctions = int(rng.choice([100, 250, 500, 1000]))
    min_forest_size = int(rng.choice([1, 2, 5]))

    return {
        # XGBoost params
        "xgb_n_estimators": xgb_n_estimators,
        "xgb_max_depth": xgb_max_depth,
        "xgb_learning_rate": xgb_learning_rate,
        "xgb_colsample_bytree": xgb_colsample_bytree,
        "xgb_subsample": xgb_subsample,
        "xgb_min_child_weight": xgb_min_child_weight,
        "xgb_reg_lambda": xgb_reg_lambda,
        "xgb_reg_alpha": xgb_reg_alpha,
        "xgb_gamma": xgb_gamma,
        # FBT params
        "max_depth": max_depth,
        "max_number_of_conjunctions": max_number_of_conjunctions,
        "min_forest_size": min_forest_size,
    }


def prepare_and_save_results(results, args):
    """
    Prepare all data for serialization and save to JSON file.

    Args:
        results: Dictionary of results (raw values)
        args: Command line arguments
    """
    serialized_results = {}
    for key, value in results.items():
        serialized_results[key] = f"{value:.6f}"

    output = {
        "model": "FBT",
        "trial_id": str(args.trial_id),
        "fold": str(args.fold),
        "dataset": args.dataset,
        "random_seed": str(args.random_seed),
        "base_seed": str(args.base_seed),
    }
    output.update(serialized_results)

    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(
        output_path,
        f"fbt_results_trial_{args.dataset}_{args.trial_id}_{args.fold}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FBT Hyperparameter Trial Runner"
    )
    parser.add_argument(
        "--dataset", type=str, default="room", help="Data name. Default is room"
    )
    parser.add_argument(
        "--trial-id",
        type=int,
        required=True,
        help="Trial ID (typically from SLURM_ARRAY_TASK_ID)",
    )
    parser.add_argument(
        "--base-seed", type=int, default=42, help="Base random seed"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60 * 60,
        help="Timeout for each trial in seconds",
    )
    parser.add_argument(
        "--home-dir", type=str, default="./", help="Home directory for data and results"
    )
    parser.add_argument(
        "--fold", type=int, default=0, help="Data fold to use, default is 0"
    )
    parser.add_argument(
        "--min_time",
        type=int,
        default=3,
        help="Minimum time for each trial in seconds",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel threads for XGBoost base model (-1 for all cores)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for XGBoost base model: 'cpu' or 'cuda'",
    )
    args = parser.parse_args()
    args.output_dir = os.path.join(args.home_dir, "results")

    t0 = time.time()

    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)

    data_factory = DataFactory_clf(
        args.dataset, cache_dir=os.path.join(args.home_dir, "data")
    )

    hyperparams = sample_hyperparameters(rng)
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(args.fold)

    # FBT.fit() expects a pandas DataFrame, same as ShapeFBT
    feature_cols = [f"f{i}" for i in range(X_train.shape[1])]
    label_col = "label"

    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[label_col] = y_train

    val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_df[label_col] = y_val

    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[label_col] = y_test

    # ── Stage 1: fit XGBoost base model ───────────────────────────────────
    print("constructing XGBoost base model")
    xgb_model = XGBClassifier(
        n_estimators=hyperparams["xgb_n_estimators"],
        max_depth=hyperparams["xgb_max_depth"],
        learning_rate=hyperparams["xgb_learning_rate"],
        colsample_bytree=hyperparams["xgb_colsample_bytree"],
        subsample=hyperparams["xgb_subsample"],
        min_child_weight=hyperparams["xgb_min_child_weight"],
        reg_lambda=hyperparams["xgb_reg_lambda"],
        reg_alpha=hyperparams["xgb_reg_alpha"],
        gamma=hyperparams["xgb_gamma"],
        random_state=args.random_seed,
        nthread=args.n_jobs,
        device=args.device,
        eval_metric="mlogloss",
        verbosity=0,
    )

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(args.timeout)
    print("fitting XGBoost base model")
    try:
        t1 = time.time()
        xgb_model.fit(X_train, y_train)

        # ── Stage 2: fit FBT on top of XGBoost ────────────────────────────
        print("constructing FBT")
        clf = FBT(
            max_depth=hyperparams["max_depth"],
            min_forest_size=hyperparams["min_forest_size"],
            max_number_of_conjunctions=hyperparams["max_number_of_conjunctions"],
        )

        print("fitting FBT")
        clf.fit(
            train=train_df,
            feature_cols=feature_cols,
            label_col=label_col,
            xgb_model=xgb_model,
        )
        signal.alarm(0)
    except TimeoutException:
        signal.alarm(0)
        results = {
            "train_acc": float("nan"),
            "val_acc": float("nan"),
            "test_acc": float("nan"),
            "elapsed_time": args.timeout,
        }
        prepare_and_save_results(results, args)
        t3 = time.time()
        final_elapsed = t3 - t0
        if final_elapsed < args.min_time:
            time.sleep(args.min_time - final_elapsed)
        sys.exit(124)
    print("done fitting FBT")

    t2 = time.time()
    elapsed_time = t2 - t1

    print("predicting")
    train_pred = clf.predict(train_df[feature_cols])
    val_pred = clf.predict(val_df[feature_cols])
    test_pred = clf.predict(test_df[feature_cols])
    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)

    results = {
        "train_acc": train_acc,
        "val_acc": val_acc,
        "test_acc": test_acc,
        "elapsed_time": elapsed_time,
    }
    print("done predicting")
    print("saving results")
    prepare_and_save_results(results, args)
    print("done saving results")
    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)