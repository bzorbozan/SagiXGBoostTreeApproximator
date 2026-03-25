from data_utils import *
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import signal
import time
import json
import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def sample_hyperparameters(rng):
    """
    Sample a set of XGBoost hyperparameters.

    Returns:
        dict: Sampled hyperparameters
    """
    # Max depth of each tree
    max_depth = int(rng.integers(2,7))
 
    # Number of boosting rounds
    n_estimators = int(rng.choice([50, 100, 250, 500]))
 
    # Learning rate (step size shrinkage)
    learning_rate = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
 
    # Fraction of features to use per tree (colsample_bytree)
    colsample_bytree = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
 
    # Fraction of training samples to use per tree
    subsample = float(rng.choice([0.5, 0.63, 0.8, 1.0]))
 
    # Minimum sum of instance weight in a child node
    min_child_weight = int(rng.choice([1, 3, 5, 10]))
 
    # L2 regularization term on weights
    reg_lambda = float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0]))
 
    # L1 regularization term on weights
    reg_alpha = float(rng.choice([0.0, 0.1, 0.5, 1.0]))
 
    # Minimum loss reduction required to make a further partition
    gamma = float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))
 
    return {
        "max_depth": max_depth,
        "n_estimators": n_estimators,
        "learning_rate": learning_rate,
        "colsample_bytree": colsample_bytree,
        "subsample": subsample,
        "min_child_weight": min_child_weight,
        "reg_lambda": reg_lambda,
        "reg_alpha": reg_alpha,
        "gamma": gamma,
    }
def prepare_and_save_results(results, args, hyperparams):
    """
    Prepare all data for serialization and save to JSON file.
 
    Args:
        results: Dictionary of results (raw values)
        args: Command line arguments
        hyperparams: Sampled hyperparameters (for logging)
    """
    serialized_results = {}
    for key, value in results.items():
        serialized_results[key] = f"{value:.6f}"
 
    output = {
        "model": "XGBoost",
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
        f"xgb_results_trial_{args.dataset}_{args.trial_id}_{args.fold}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XGBoost Hyperparameter Trial Runner"
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
        "--home-dir", type=str, default="./", help="Output directory for results"
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
        help="Number of parallel threads for XGBoost (-1 for all cores)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for XGBoost: 'cpu' or 'cuda'",
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
 
    print("constructing classifier")
    clf = XGBClassifier(
        n_estimators=hyperparams["n_estimators"],
        max_depth=hyperparams["max_depth"],
        learning_rate=hyperparams["learning_rate"],
        colsample_bytree=hyperparams["colsample_bytree"],
        subsample=hyperparams["subsample"],
        min_child_weight=hyperparams["min_child_weight"],
        reg_lambda=hyperparams["reg_lambda"],
        reg_alpha=hyperparams["reg_alpha"],
        gamma=hyperparams["gamma"],
        random_state=args.random_seed,
        nthread=args.n_jobs,
        device=args.device,
        eval_metric="mlogloss", # this is the default for classifiers
        verbosity=0,
    )
 
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(args.timeout)
    print("fitting classifier")
    try:
        t1 = time.time()
        clf.fit(X_train, y_train)
        signal.alarm(0)
    except TimeoutException:
        signal.alarm(0)
        results = {
            "train_acc": float("nan"),
            "val_acc": float("nan"),
            "test_acc": float("nan"),
            "elapsed_time": args.timeout,
        }
        prepare_and_save_results(results, args, hyperparams)
        t3 = time.time()
        final_elapsed = t3 - t0
        if final_elapsed < args.min_time:
            time.sleep(args.min_time - final_elapsed)
        sys.exit(124)
    print("done fitting classifier")
 
    t2 = time.time()
    elapsed_time = t2 - t1
 
    print("predicting")
    train_pred = clf.predict(X_train)
    val_pred = clf.predict(X_val)
    test_pred = clf.predict(X_test)
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
    prepare_and_save_results(results, args, hyperparams)
    print("done saving results")
    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)