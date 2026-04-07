"""
Aggregate experiment JSON results into summary CSVs with Hyperparameter Reconstruction.

This script performs a fresh scan of specific result directories to build a 
comprehensive dataset of experiment runs. It uniquely reconstructs the 
hyperparameters for each trial by replaying the original Random Number 
Generator (RNG) logic used during the experiment.

Key Features & Logic:
1. Directory-to-Prefix Mapping:
   - Scans designated subfolders under 'all_results/' for specific file prefixes:
     * 'all_results/xgb_shapefbt_results' -> [xgb_results, shapefbt_results]
     * 'all_results/sagi_results'         -> [fbt_results]
     * 'all_results/shapecart_results'    -> [shapecart_results]

2. Hyperparameter Reconstruction (RNG Replay):
   - Uses 'base_seed' + 'trial_id' from each JSON to recreate the exact 
     numpy.random.Generator used during the original run.
   - Replays the specific sampling logic for XGBoost, ShapeFBT, FBT, and 
     ShapeCART to recover the original hyperparameter configurations.

3. Metric-Based Success Filtering:
   - Success is determined by the presence of valid numeric accuracy metrics. 
   - Automatically filters out files containing "nan" or missing 'train_acc', 
     'val_acc', or 'test_acc'.

4. Comprehensive Metadata & Duplicate Reporting:
   - Identifies duplicates based on [dataset, model, trial_id, max_depth, fold] 
     and writes details to 'all_results_duplicates.csv'.
   - Calculates 'generalization_gap' (|train_acc - test_acc|) for every run.
   - Creates an '_agg_trial' key (trial_id + max_depth) specifically for ShapeCART.

5. Statistical Aggregation:
   - Calculates Mean and Standard Deviation for all metrics across trials.
   - Identifies the "winner" (best val_acc) per model/dataset.
   - Computes global model averages across all datasets.

Outputs (Saved to /processed_results):
- results_with_hyperparameters.csv: The master table containing all 15 metadata 
  columns, accuracy metrics, file paths, fingerprints, and the reconstructed 
  hyperparameters (as a JSON string).
- best_per_dataset.csv / _std.csv: Winning trial metrics (Mean and SD).
- best_avg.csv / _std.csv: High-level model comparison (Mean and SD).
- all_results_duplicates.csv: Audit list of flagged duplicate files.
"""

import argparse
import json
import os
import re
import pandas as pd
import numpy as np

# Folder to Prefix Mapping
FOLDER_MAP = {
    "all_results/xgb_shapefbt_results": ["xgb_results", "shapefbt_results"],
    "all_results/sagi_results": ["fbt_results"],
    "all_results/shapecart_results": ["shapecart_results"]
}

def _normalize_model_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

# --- Hyperparameter Sampling Logic (Replay) ---

def sample_shapecart_hp(rng):
    """Replays the ShapeCART Optuna-style search space."""
    return {
        'criterion': str(rng.choice(['gini', 'entropy'])),
        'min_samples_split': int(2 ** rng.integers(1, 6)),
        'min_samples_leaf': int(rng.integers(1, 33)),
        'min_impurity_decrease': float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01])),
        'inner_max_leaf_nodes': int(rng.choice([x * 4 for x in range(1, 17)])),
        'inner_min_samples_leaf': rng.choice([1, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    }

def sample_shapefbt_hp(rng):
    """Replays the ShapeFBT specific sampler from shapefbt_run_new.py."""
    return {
        "xgb_n_estimators": int(rng.choice([50, 100, 250, 500])),
        "xgb_max_depth": int(rng.integers(2, 7)),
        "xgb_learning_rate": float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3])),
        "outer_tree_max_depth": int(rng.integers(2, 7)),
        "inner_tree_max_depth": int(rng.integers(2, 7)),
        "max_number_of_conjunctions": int(rng.choice([100, 250, 500, 1000])),
        "min_forest_size": int(rng.choice([1, 2, 5])),
        "k": 2,
        "min_samples_split": int(rng.choice([5, 10, 20])),
        "inner_tree_criterion": str(rng.choice(['entropy', 'gini']))
    }

def sample_xgb_hp(rng):
    """Replays XGBoost sampler from xgb_run_new.py."""
    return {
        "max_depth": int(rng.integers(2, 7)),
        "n_estimators": int(rng.choice([50, 100, 250, 500])),
        "learning_rate": float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3])),
        "colsample_bytree": float(rng.choice([0.25, 0.5, 0.75, 1.0])),
        "subsample": float(rng.choice([0.5, 0.63, 0.8, 1.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))
    }

def sample_fbt_hp(rng):
    """Replays SagiFBT/FBT sampler from sagifbt_run_new.py."""
    return {
        "xgb_max_depth": int(rng.integers(3, 9)),
        "max_depth": int(rng.integers(2, 7)),
        "max_number_of_conjunctions": int(rng.choice([100, 250, 500, 1000])),
        "min_forest_size": int(rng.choice([1, 2, 5]))
    }

def reconstruct_hp(data):
    """Reconstructs hyperparams using base_seed + trial_id."""
    try:
        model = str(data.get("model", ""))
        base_seed = int(data["base_seed"])
        trial_id = int(data["trial_id"])
        
        trial_seed = base_seed + trial_id
        rng = np.random.default_rng(trial_seed)
        
        if "ShapeCART" in model:
            hp = sample_shapecart_hp(rng)
            if data.get("max_depth"): hp["max_depth"] = data["max_depth"]
            return hp
        elif "ShapeFBT" in model:
            return sample_shapefbt_hp(rng)
        elif "XGBoost" in model:
            return sample_xgb_hp(rng)
        elif "FBT" in model:
            return sample_fbt_hp(rng)
    except Exception:
        return {}
    return {}

# --- Processing & Aggregation Logic ---

def check_duplicates(df_new, output_dir):
    """Identifies and logs duplicates to all_results_duplicates.csv."""
    duplicate_log = []
    id_cols = ['dataset', 'model', 'trial_id', 'max_depth', 'fold']
    existing_id_cols = [c for c in id_cols if c in df_new.columns]
    
    id_dupes = df_new[df_new.duplicated(subset=existing_id_cols, keep=False)]
    for _, row in id_dupes.iterrows():
        duplicate_log.append({
            "type": "ID DUPE", "model": row.get('model'), "dataset": row.get('dataset'),
            "trial_id": row.get('trial_id'), "max_depth": row.get('max_depth'),
            "fold": row.get('fold'), "file_path": row.get('file_path')
        })

    if duplicate_log:
        dupe_df = pd.DataFrame(duplicate_log).drop_duplicates()
        dupe_df.to_csv(os.path.join(output_dir, "all_results_duplicates.csv"), index=False)
        print(f"⚠️  Found {len(dupe_df)} duplicate entries.")

def load_results():
    """Walks folders, loads JSONs, performs RNG replay, and calculates metadata."""
    results = []
    for folder, prefixes in FOLDER_MAP.items():
        if not os.path.exists(folder): continue
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json") and any(file.startswith(p) for p in prefixes):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                        
                        # --- 1. Basic Metadata & Accuracy ---
                        data['file_path'] = path
                        data['fingerprint'] = json.dumps(data, sort_keys=True)
                        
                        # --- 2. Numeric Conversion (for consistency) ---
                        for col in ["train_acc", "val_acc", "test_acc", "elapsed_time", "max_depth", "trial_id", "fold"]:
                            if col in data:
                                try:
                                    data[col] = float(data[col])
                                except (ValueError, TypeError):
                                    data[col] = np.nan

                        # Skip if metrics are invalid
                        if pd.isna(data.get('train_acc')) or pd.isna(data.get('test_acc')):
                            continue

                        # --- 3. Calculated Columns ---
                        # Generalization Gap
                        data['generalization_gap'] = abs(data['train_acc'] - data['test_acc'])
                        
                        # Aggregated Trial ID (ShapeCART logic)
                        norm_m = _normalize_model_name(data.get('model', ''))
                        if 'shapecart' in norm_m:
                            t_id = int(data.get('trial_id', 0))
                            m_depth = int(data.get('max_depth', 0))
                            data['_agg_trial'] = f"{t_id}_{m_depth}"
                        else:
                            data['_agg_trial'] = str(data.get('trial_id', ''))

                        # --- 4. Hyperparameter Replay ---
                        hp = reconstruct_hp(data)
                        data['hyperparameters'] = json.dumps(hp, sort_keys=True)
                        
                        results.append(data)
                    except Exception: continue

    if not results: return pd.DataFrame()
    
    # Create DataFrame and ensure column order matches your request
    df = pd.DataFrame(results)
    
    # Define the specific column order you requested
    requested_columns = [
        'model', 'trial_id', 'fold', 'dataset', 'random_seed', 'base_seed',
        'train_acc', 'val_acc', 'test_acc', 'elapsed_time', 'file_path',
        'fingerprint', 'max_depth', '_agg_trial', 'generalization_gap', 'hyperparameters'
    ]
    
    # Filter to columns that actually made it into the DF (to avoid errors)
    final_columns = [c for c in requested_columns if c in df.columns]
    
    return df[final_columns]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="processed_results")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results()
    if df.empty: exit("No valid results found.")

    check_duplicates(df, args.output_dir)

    # Aggregated Trial ID logic for ShapeCART
    is_shapecart = df["model"].astype(str).apply(_normalize_model_name).str.contains("shapecart")
    df["_agg_trial"] = df["trial_id"].astype(str)
    if "max_depth" in df.columns:
        df.loc[is_shapecart, "_agg_trial"] = (
            df.loc[is_shapecart, "trial_id"].fillna(0).astype(int).astype(str) + "_" + 
            df.loc[is_shapecart, "max_depth"].fillna(0).astype(int).astype(str)
        )

    # Statistical Aggregation
    metrics = ["train_acc", "val_acc", "test_acc", "elapsed_time"]
    df_mean = df.groupby(["model", "dataset", "_agg_trial"])[metrics].mean().reset_index()
    df_std = df.groupby(["model", "dataset", "_agg_trial"])[metrics].std().reset_index()
    
    best_idxs = df_mean.groupby(["dataset", "model"])["val_acc"].idxmax()
    df_best = df_mean.loc[best_idxs].reset_index(drop=True)
    df_best_std = df_std.loc[best_idxs].reset_index(drop=True)
    
    df_best_avg = df_best.groupby("model")[metrics].mean().round(3).reset_index()
    df_best_avg_std = df_best_std.groupby("model")[metrics].mean().round(3).reset_index()

    # Final Output
    outputs = {
        "results_with_hyperparameters.csv": df,
        "best_per_dataset.csv": df_best,
        "best_per_dataset_std.csv": df_best_std,
        "best_avg.csv": df_best_avg,
        "best_avg_std.csv": df_best_avg_std
    }

    for name, data in outputs.items():
        data.to_csv(os.path.join(args.output_dir, name), index=False)
    
    print(f"✅ Success! All tables generated in {args.output_dir}")