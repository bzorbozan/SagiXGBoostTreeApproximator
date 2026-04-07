import signal
import time
import json
import argparse
import numpy as np
import sys
import os
import pandas as pd 

# Environment setup
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "src"))

from data_utils import *
from xgboost import XGBClassifier
from FBT import FBT
from sklearn.metrics import accuracy_score

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def prepare_and_save_results(results, args): 
    """
    Saves essential parameters and metrics. Filename includes n_estimators 
     and max_depth to ensure uniqueness.
    """
    serialized_results = {}
    for key, value in results.items():
        serialized_results[key] = f"{value:.6f}" if isinstance(value, float) else value
 
    output = {
        "model": "FBT",
        "trial_id": str(args.trial_id),
        "fold": str(args.fold),
        "dataset": args.dataset,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        **serialized_results
    }
 
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)
 
    # Updated naming convention to prevent clobbering
    output_file = os.path.join(
        output_path,
        f"fbt_capability_{args.dataset}_n{args.n_estimators}_d{args.max_depth}_trial_{args.trial_id}_fold_{args.fold}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FBT Deterministic Trial Runner")
    parser.add_argument("--dataset", type=str, default="room")
    parser.add_argument("--trial-id", type=int, required=True)
    parser.add_argument("--base-seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument("--home-dir", type=str, default="./")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--min_time", type=int, default=3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--device", type=str, default="cpu")
    
    # Grid variables
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=6)

    args = parser.parse_args()
    
    args.output_dir = os.path.join(args.home_dir, "results_fbt_capability")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Define pinned hyperparameters for FBT and XGBoost
    hyperparams = {
        # XGBoost (Stage 1) - Pinned values to match your specific study requirements
        "xgb_n_estimators": args.n_estimators,
        "xgb_max_depth": args.max_depth,
        "xgb_learning_rate": 0.1,
        "xgb_colsample_bytree": 1.0,
        "xgb_subsample": 1.0,
        "xgb_min_child_weight": 1,
        "xgb_reg_lambda": 0.0,
        "xgb_reg_alpha": 0.0,
        "xgb_gamma": 0.0,

        # FBT (Stage 2) - Pinned values (based on your FBT reference logic)
        "max_depth": 6,
        "max_number_of_conjunctions": 500,
        "min_forest_size": 5,
    }
 
    t0 = time.time()
    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)
 
    data_factory = DataFactory_clf(args.dataset, cache_dir=os.path.join(args.home_dir, "data"))
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(args.fold)
 
    feature_cols = [f"f{i}" for i in range(X_train.shape[1])]
    label_col = "label"
 
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[label_col] = y_train
    val_df = pd.DataFrame(X_val, columns=feature_cols)
    val_df[label_col] = y_val
    test_df = pd.DataFrame(X_test, columns=feature_cols)
    test_df[label_col] = y_test
    
    print(f"Constructing XGBoost (n={args.n_estimators}, d={args.max_depth})")
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

    try:
        t1 = time.time()
        print("Fitting XGBoost base model")
        xgb_model.fit(X_train, y_train)
 
        print("Constructing FBT")
        clf = FBT(
            max_depth=hyperparams["max_depth"],
            min_forest_size=hyperparams["min_forest_size"],
            max_number_of_conjunctions=hyperparams["max_number_of_conjunctions"],
        )
 
        print("Fitting FBT")
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
            "elapsed_time": float(args.timeout),
        }
        prepare_and_save_results(results, args)
        t3 = time.time()
        final_elapsed = t3 - t0
        if final_elapsed < args.min_time:
            time.sleep(args.min_time - final_elapsed)
        sys.exit(124)
 
    t2 = time.time()
    elapsed_time = t2 - t1
 
    print("Predicting")
    train_pred = clf.predict(train_df[feature_cols])
    val_pred = clf.predict(val_df[feature_cols])
    test_pred = clf.predict(test_df[feature_cols])

    results = {
        "train_acc": accuracy_score(y_train, train_pred),
        "val_acc": accuracy_score(y_val, val_pred),
        "test_acc": accuracy_score(y_test, test_pred),
        "elapsed_time": elapsed_time,
    }

    print("Saving results")
    prepare_and_save_results(results, args)
    
    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)
    print("Done")