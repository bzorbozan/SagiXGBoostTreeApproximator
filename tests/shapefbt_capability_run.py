import signal
import time
import json
import argparse
import numpy as np
import sys
import os
import pandas as pd 

# before model files were moved into a src folder
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# after moving model files into src - not used to run the full thing , done after - make sure u test if u wanna run again
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(_root)
sys.path.append(os.path.join(_root, "src"))

from data_utils import *
from xgboost import XGBClassifier
from ShapeFBT import ShapeFBT
from sklearn.metrics import accuracy_score

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

# We no longer have the sample_hyperparameters function, because we are running a deterministic search. hyperparameters are defined below

def prepare_and_save_results(results, args): 
    """
    Saves only the essential grid parameters to minimize JSON file size.
    """
    serialized_results = {}
    for key, value in results.items():
        # Keep 6 decimal places for floats, otherwise keep as is
        serialized_results[key] = f"{value:.6f}" if isinstance(value, float) else value
 
    output = {
        "model": "ShapeFBT",
        "trial_id": str(args.trial_id),
        "fold": str(args.fold),
        "dataset": args.dataset,
        # Only saving the variables that change
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        **serialized_results
    }
 
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)
 
    output_file = os.path.join(
        output_path,
        # f"shapefbt_capability_trial_{args.trial_id}_fold_{args.fold}.json"
        # Update the filename to include the grid parameters (in case trial id is not unique)
        f"shapefbt_capability_{args.dataset}_n{args.n_estimators}_d{args.max_depth}_trial_{args.trial_id}_fold_{args.fold}.json",
    )
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ShapeFBT Hyperparameter Trial Runner"
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
        "--home-dir",
        type=str,
        default=_root,
        help="Project root directory for data and results",
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
    parser.add_argument(
        "--n-estimators", 
        type=int, 
        default=100, 
        help="Number of XGBoost estimators for this specific run"
    )
    parser.add_argument(
        "--max-depth", 
        type=int, 
        default=6,
        help='Max depth for XGBoost base model')

    args = parser.parse_args()

    hyperparams = {
        # XGBoost Pinned + Variable
        "xgb_n_estimators": args.n_estimators, # From CLI
        "xgb_max_depth": args.max_depth,       # From CLI
        "xgb_learning_rate": 0.1,              # Pinned
        "xgb_colsample_bytree": 1.0,           # Pinned
        "xgb_subsample": 1.0,                  # Pinned
        "xgb_min_child_weight": 1,             # Pinned
        "xgb_reg_lambda": 0.0,
        "xgb_reg_alpha": 0.0,
        "xgb_gamma": 0.0,

        # ShapeFBT Pinned (Matches your 'make_shapefbt' requirements)
        "outer_tree_max_depth": 6,
        "inner_tree_max_depth": 6,
        "max_number_of_conjunctions": 500,
        "min_forest_size": 5,
        "k": 2,
        "min_samples_split": 20,
        "min_conjunctions_split": 2,
        "min_impurity_decrease": 0.0,
        "inner_tree_criterion": "entropy",
    }
    
    args.output_dir = os.path.join(args.home_dir, "results")
 
    t0 = time.time()
 
    trial_seed = args.base_seed + args.trial_id
    args.random_seed = trial_seed
    rng = np.random.default_rng(trial_seed)
 
    data_factory = DataFactory_clf(
        args.dataset, cache_dir=os.path.join(args.home_dir, "data")
    )
 
    # hyperparams = sample_hyperparameters(rng)
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(args.fold)
 
    # ADDITION - CONVERT INPUT TO PD DATAFRAME
    # data_utils returns numpy arrays; ShapeFBT.fit() expects a pandas DataFrame !!!!!

    # feature_cols = [str(i) for i in range(X_train.shape[1])] --> this is not the naming convention shapefbt expects
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
        eval_metric="mlogloss", # this is the default for classifiers
        verbosity=0,
    )
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(args.timeout)
    print("fitting XGBoost base model")
    try:
        t1 = time.time()
        xgb_model.fit(X_train, y_train)
 
        # ── Stage 2: fit ShapeFBT on top of XGBoost ───────────────────────
        print("constructing ShapeFBT")
        clf = ShapeFBT(
            outer_tree_max_depth=hyperparams["outer_tree_max_depth"],
            inner_tree_max_depth=hyperparams["inner_tree_max_depth"],
            min_forest_size=hyperparams["min_forest_size"],
            max_number_of_conjunctions=hyperparams["max_number_of_conjunctions"],
            k=hyperparams["k"],
            min_samples_split=hyperparams["min_samples_split"],
            min_conjunctions_split=hyperparams["min_conjunctions_split"],
            min_impurity_decrease=hyperparams["min_impurity_decrease"],
            inner_tree_criterion=hyperparams["inner_tree_criterion"],
        )
 
        print("fitting ShapeFBT")
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
        prepare_and_save_results(results, args) # results, args, hyperparams
        t3 = time.time()
        final_elapsed = t3 - t0
        if final_elapsed < args.min_time:
            time.sleep(args.min_time - final_elapsed)
        sys.exit(124)
    print("done fitting ShapeFBT")
 
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
    prepare_and_save_results(results, args) # results, args, hyperparams
    print("done saving results")
    t3 = time.time()
    final_elapsed = t3 - t0
    if final_elapsed < args.min_time:
        time.sleep(args.min_time - final_elapsed)
    print(f"Trial {args.trial_id} with {args.max_depth} max depth and {args.n_estimators} n_estimators completed. Test Acc: {test_acc:.4f} | Time: {elapsed_time:.2f}s")