# FAST SETUP — run once before slow cells
import os
import sys
import time
import random
import traceback
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
try:
    import resource
except Exception:
    resource = None
if os.getcwd().endswith('/tests'):
    repo_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
else:
    repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.append(repo_root)
from FBT import FBT
from tests.data_utils import DataFactory_clf
print('OK:', repo_root)


# Historic OOM trial IDs (from prior XGB-approx experiments) replayed for SagiFBT.
# Run cell 1 first. Same seed convention as tests/sagifbt_run_new.py:
#   trial_seed = base_seed + trial_id
# Default base_seed=42 matches that script's --base-seed; change if your jobs used another base.

def sample_hyperparams(rng, force_xgb_n_estimators=1000):
    """Same RNG draw order as tests/sagifbt_run_new.sample_hyperparameters; n_estimators sampled then replaced with force_xgb_n_estimators."""
    xgb_max_depth = int(rng.integers(3, 9))
    _ = int(rng.choice([50, 100, 250, 500, 750, 1000]))
    xgb_learning_rate = float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3]))
    xgb_colsample_bytree = float(rng.choice([0.25, 0.5, 0.75, 1.0]))
    xgb_subsample = float(rng.choice([0.5, 0.63, 0.8, 1.0]))
    xgb_min_child_weight = int(rng.choice([1, 3, 5, 10]))
    xgb_reg_lambda = float(rng.choice([0.0, 0.1, 1.0, 5.0, 10.0]))
    xgb_reg_alpha = float(rng.choice([0.0, 0.1, 0.5, 1.0]))
    xgb_gamma = float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))
    max_depth = int(rng.integers(2, 7))
    max_number_of_conjunctions = int(rng.choice([100, 250, 500, 1000]))
    min_forest_size = int(rng.choice([1, 2, 5]))
    return {
        "xgb_n_estimators": force_xgb_n_estimators,
        "xgb_max_depth": xgb_max_depth,
        "xgb_learning_rate": xgb_learning_rate,
        "xgb_colsample_bytree": xgb_colsample_bytree,
        "xgb_subsample": xgb_subsample,
        "xgb_min_child_weight": xgb_min_child_weight,
        "xgb_reg_lambda": xgb_reg_lambda,
        "xgb_reg_alpha": xgb_reg_alpha,
        "xgb_gamma": xgb_gamma,
        "max_depth": max_depth,
        "max_number_of_conjunctions": max_number_of_conjunctions,
        "min_forest_size": min_forest_size,
    }
# HISTORIC_TRIAL_IDS = [1, 2, 3, 4, 5, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25, 29, 30, 31, 32, 33, 36, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 55, 58, 60, 61, 62, 64, 67, 68, 69, 71, 73, 75, 77, 78, 79, 80, 81, 83, 84, 85, 86, 87, 88, 89, 90, 91, 93, 94, 95, 98, 99, 100, 102, 103, 104, 107, 109, 110, 111, 112, 114, 116, 119, 120, 121, 122, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 144, 145, 147, 148, 149, 150, 151, 152, 153, 154, 155, 159, 160, 163, 166, 167, 168, 169, 171, 173, 174, 175, 178, 179, 180, 181, 182, 183, 185, 186, 187, 189, 190, 193, 194, 195, 196, 197, 198, 199, 200]
HISTORIC_TRIAL_IDS = [1, 2, 3, 4, 5, 10]
dataset = "room"
folds = [0, 1, 2, 3, 4]
base_seed = 42  # match tests/sagifbt_run_new.py default; override if your SLURM jobs differ
use_device = "cpu"
n_jobs = -1
stop_on_memory_error = False  # True: stop entire grid on first MemoryError

results_ok = []
results_mem = []
results_err = []

for fold in folds:
    data_factory = DataFactory_clf(dataset, cache_dir=os.path.join(repo_root, 'data'))
    X_train, y_train, X_val, y_val, X_test, y_test = data_factory.get_data(fold)
    feature_cols = [f'f{i}' for i in range(X_train.shape[1])]
    label_col = 'label'
    train_df = pd.DataFrame(X_train, columns=feature_cols)
    train_df[label_col] = y_train

    for trial_id in HISTORIC_TRIAL_IDS:
        trial_seed = base_seed + trial_id
        rng = np.random.default_rng(trial_seed)
        hp = sample_hyperparams(rng, force_xgb_n_estimators=1000)

        print(f"\n=== fold={fold} trial_id={trial_id} trial_seed={trial_seed} ===")
        for k, v in hp.items():
            print(f"  {k}: {v}")

        xgb_model = XGBClassifier(
            n_estimators=hp["xgb_n_estimators"],
            max_depth=hp["xgb_max_depth"],
            learning_rate=hp["xgb_learning_rate"],
            colsample_bytree=hp["xgb_colsample_bytree"],
            subsample=hp["xgb_subsample"],
            min_child_weight=hp["xgb_min_child_weight"],
            reg_lambda=hp["xgb_reg_lambda"],
            reg_alpha=hp["xgb_reg_alpha"],
            gamma=hp["xgb_gamma"],
            random_state=trial_seed,
            nthread=n_jobs,
            device=use_device,
            eval_metric="mlogloss",
            verbosity=0,
        )
        clf = FBT(
            max_depth=hp["max_depth"],
            min_forest_size=hp["min_forest_size"],
            max_number_of_conjunctions=hp["max_number_of_conjunctions"],
        )

        t0 = time.time()
        try:
            xgb_model.fit(X_train, y_train)
            clf.fit(
                train=train_df,
                feature_cols=feature_cols,
                label_col=label_col,
                xgb_model=xgb_model,
            )
        except MemoryError:
            print(f"MemoryError fold={fold} trial_id={trial_id} after {time.time() - t0:.2f}s")
            results_mem.append((fold, trial_id))
            if stop_on_memory_error:
                raise
        except Exception:
            print(f"Exception fold={fold} trial_id={trial_id} after {time.time() - t0:.2f}s:")
            traceback.print_exc()
            results_err.append((fold, trial_id))
        else:
            dt = time.time() - t0
            print(f"OK in {dt:.2f}s")
            results_ok.append((fold, trial_id, dt))
            if resource is not None:
                print(
                    "Peak RSS (platform-dependent units):",
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                )

print("\n--- summary ---")
print("ok:", len(results_ok), "MemoryError:", len(results_mem), "other errors:", len(results_err))
if results_mem:
    tail = " ..." if len(results_mem) > 20 else ""
    print("MemoryError (fold, trial_id):", results_mem[:20], tail)
