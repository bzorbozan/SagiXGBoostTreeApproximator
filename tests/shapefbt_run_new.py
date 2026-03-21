from data_utils import *
from xgboost import XGBClassifier
from ShapeFBT import ShapeFBT
from sklearn.metrics import accuracy_score
import signal
import time
import json
import argparse
import numpy as np
import sys
import os
import pandas as pd # is this allowed????

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def sample_hyperparameters(rng):
    """
    Sample a set of ShapeFBT hyperparameters.
 
    Stage 1 (XGBoost base model) and Stage 2 (ShapeFBT outer tree)
    hyperparameters are sampled together, since ShapeFBT.fit() requires
    a pre-trained XGBoost model as input.
 
    Returns:
        dict: Sampled hyperparameters
    """
    # ── XGBoost base model (stage 1) ──────────────────────────────────────
    # NOTE: I added more hyperparams for XGB (to match the other one)

    # Max depth of each tree
    xgb_max_depth = int(rng.integers(3, 9))
 
    # Number of boosting rounds
    xgb_n_estimators = int(rng.choice([50, 100, 250, 500, 750, 1000]))
 
    # Learning rate (step size shrinkage)
    xgb_learning_rate = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
 
    # Fraction of features to use per tree (colsample_bytree)
    xgb_colsample_bytree = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
 
    # Fraction of training samples to use per tree
    xgb_subsample = float(rng.choice([0.5, 0.63, 0.8, 1.0]))
 
    # Minimum sum of instance weight in a child node
    xgb_min_child_weight = int(rng.choice([1, 3, 5, 10]))
 
    # L2 regularization term on weights
    xgb_reg_lambda = float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0]))
 
    # L1 regularization term on weights
    xgb_reg_alpha = float(rng.choice([0.0, 0.1, 0.5, 1.0]))
 
    # Minimum loss reduction required to make a further partition
    xgb_gamma = float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))

    # ── ShapeFBT outer tree (stage 2) ─────────────────────────────────────
    outer_tree_max_depth = int(rng.integers(2, 7))
    inner_tree_max_depth = int(rng.integers(2, 7))
    max_number_of_conjunctions = int(rng.choice([100, 250, 500, 1000]))
    min_forest_size = int(rng.choice([1, 2, 5]))
    k = 2 # for now my model only works for k=2, but in the future can test k = int(rng.choice([2, 3])) 
    min_samples_split = int(rng.choice([5, 10, 20]))
    min_conjunctions_split = int(rng.choice([2, 5, 10]))
    min_impurity_decrease = float(rng.choice([0.0, 0.001, 0.01]))
    inner_tree_criterion = str(rng.choice(['entropy', 'gini']))

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

        # ShapeFBT params
        "outer_tree_max_depth": outer_tree_max_depth,
        "inner_tree_max_depth": inner_tree_max_depth,
        "max_number_of_conjunctions": max_number_of_conjunctions,
        "min_forest_size": min_forest_size,
        "k": k,
        "min_samples_split": min_samples_split,
        "min_conjunctions_split": min_conjunctions_split,
        "min_impurity_decrease": min_impurity_decrease,
        "inner_tree_criterion": inner_tree_criterion,
    }

def prepare_and_save_results(results, args): # (results, args, hyperparams)
    """
    Prepare all data for serialization and save to JSON file.
 
    Args:
        results: Dictionary of results (raw values)
        args: Command line arguments
    NOTE: due to the hight number of them might wanna add --> hyperparams: Sampled hyperparameters (for logging)
    """
    serialized_results = {}
    for key, value in results.items():
        serialized_results[key] = f"{value:.6f}"
 
    # hp_serialized = {k: str(v) for k, v in hyperparams.items()}
 
    output = {
        "model": "ShapeFBT",
        "trial_id": str(args.trial_id),
        "fold": str(args.fold),
        "dataset": args.dataset,
        "random_seed": str(args.random_seed),
        "base_seed": str(args.base_seed),
    #    "hyperparams": hp_serialized,
    }
    output.update(serialized_results)
 
    output_path = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_path, exist_ok=True)
 
    output_file = os.path.join(
        output_path,
        f"shapefbt_results_trial_{args.dataset}_{args.trial_id}_{args.fold}.json",
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