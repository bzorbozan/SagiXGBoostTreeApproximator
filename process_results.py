import argparse
import json
import os
import re
import pandas as pd
import numpy as np

# Folder to Prefix Mapping (Used for JSON files: XGB, ShapeFBT, FBT)
FOLDER_MAP = {
    "all_results/xgb_shapefbt_results": ["xgb_results", "shapefbt_results"],
    "all_results/sagi_results": ["fbt_results"],
    "all_results/shapecart_results": ["shapecart_results"]
}

def _normalize_model_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())

# --- Hyperparameter Sampling Replay ---

def sample_shapecart_hp(rng):
    return {
        'criterion': str(rng.choice(['gini', 'entropy'])),
        'min_samples_split': int(2 ** rng.integers(1, 6)),
        'min_samples_leaf': int(rng.integers(1, 33)),
        'min_impurity_decrease': float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01])),
        'inner_max_leaf_nodes': int(rng.choice([x * 4 for x in range(1, 17)])),
        'inner_min_samples_leaf': rng.choice([1, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    }

def sample_cart_hp(rng):
    return {
        'criterion': str(rng.choice(['gini', 'entropy'])),
        'min_samples_split': int(2 ** rng.integers(1, 6)),
        'min_samples_leaf': int(rng.integers(1, 33)),
        'min_impurity_decrease': float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01])),
        'ccp_alpha': float(rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])),
        'max_depth': int(rng.choice([2, 3, 4, 5, 6]))
    }

def sample_shapefbt_hp(rng):
    return {
        "xgb_n_estimators": int(rng.choice([50, 100, 250, 500])),
        "xgb_max_depth": int(rng.integers(2, 7)),
        "xgb_learning_rate": float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3])),
        "outer_tree_max_depth": int(rng.integers(2, 7)),
        "inner_tree_max_depth": int(rng.integers(2, 7)),
        "max_number_of_conjunctions": int(rng.choice([100, 250, 500, 1000])),
        "min_forest_size": int(rng.choice([1, 2, 5])),
        "k": 2, "min_samples_split": int(rng.choice([5, 10, 20])),
        "inner_tree_criterion": str(rng.choice(['entropy', 'gini']))
    }

def sample_xgb_hp(rng):
    return {
        "max_depth": int(rng.integers(2, 7)),
        "n_estimators": int(rng.choice([50, 100, 250, 500])),
        "learning_rate": float(rng.choice([0.01, 0.05, 0.1, 0.2, 0.3])),
        "colsample_bytree": float(rng.choice([0.25, 0.5, 0.75, 1.0])),
        "subsample": float(rng.choice([0.5, 0.63, 0.8, 1.0])),
        "gamma": float(rng.choice([0.0, 0.1, 0.5, 1.0, 5.0]))
    }

def sample_fbt_hp(rng):
    return {
        "xgb_max_depth": int(rng.integers(3, 9)),
        "max_depth": int(rng.integers(2, 7)),
        "max_number_of_conjunctions": int(rng.choice([100, 250, 500, 1000])),
        "min_forest_size": int(rng.choice([1, 2, 5]))
    }

def reconstruct_hp(data):
    """Replays RNG for all model types based on seed + trial_id."""
    try:
        model = str(data.get("model", "")).strip()
        # Convert to float first to handle '114.0' string, then to int
        base_seed = int(float(data["base_seed"]))
        trial_id = int(float(data["trial_id"]))
        rng = np.random.default_rng(base_seed + trial_id)
        
        m_norm = model.lower()
        if "shapecart" in m_norm: return sample_shapecart_hp(rng)
        if "cart" in m_norm: return sample_cart_hp(rng)
        if "shapefbt" in m_norm: return sample_shapefbt_hp(rng)
        if "xgboost" in m_norm: return sample_xgb_hp(rng)
        if "fbt" in m_norm: return sample_fbt_hp(rng)
    except: return {}
    return {}

def process_row_metadata(data):
    """Parses accuracy and sets aggregation keys for grouping."""
    try:
        for col in ['train_acc', 'val_acc', 'test_acc', 'elapsed_time', 'max_depth', 'trial_id', 'fold', 'base_seed']:
            if col in data:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        data['generalization_gap'] = abs(data['train_acc'] - data['test_acc'])
        
        norm_m = _normalize_model_name(str(data.get('model', '')))
        t_id = int(float(data.get('trial_id', 0)))
        
        if 'shapecart' in norm_m:
            m_depth = int(float(data.get('max_depth', 0)))
            data['_agg_trial'] = f"{t_id}_{m_depth}"
        else:
            data['_agg_trial'] = str(t_id)
            
        hp = reconstruct_hp(data)
        data['hyperparameters'] = json.dumps(hp, sort_keys=True)
    except: pass
    return data

def load_results(shapecart_source, external_csv_path):
    results = []
    
    # 1. Process JSON Files
    for folder, prefixes in FOLDER_MAP.items():
        if not os.path.exists(folder): continue
        if shapecart_source == "external_csv" and "shapecart_results" in folder:
            print(f"Skipping JSON folder: {folder}")
            continue
            
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json") and any(file.startswith(p) for p in prefixes):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r") as f: data = json.load(f)
                        data['file_path'] = path
                        results.append(process_row_metadata(data))
                    except: continue

    # 2. Process external_results.csv strictly
    if os.path.exists(external_csv_path):
        print(f"--- Processing: {external_csv_path} ---")
        ext_df = pd.read_csv(external_csv_path)
        # Drop unnamed index columns
        ext_df = ext_df.loc[:, ~ext_df.columns.str.contains('^Unnamed')]
        
        targets = ["cart"]
        if shapecart_source == "external_csv":
            targets.append("shapecart")
        
        # Strip and filter models
        ext_df['model_clean'] = ext_df['model'].astype(str).str.strip().str.lower()
        mask = ext_df['model_clean'].isin(targets)
        ext_records = ext_df[mask].to_dict('records')
        
        print(f"--- Loaded {len(ext_records)} rows for {targets} from CSV ---")
        
        for row in ext_records:
            row['file_path'] = external_csv_path
            results.append(process_row_metadata(row))
    else:
        print(f"!!! Error: {external_csv_path} not found in directory !!!")

    if not results: return pd.DataFrame()
    return pd.DataFrame(results).dropna(subset=["train_acc", "val_acc", "test_acc"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="processed_results")
    parser.add_argument("--shapecart_source", choices=["zipped_files", "external_csv"], default="external_csv")
    parser.add_argument("--external_csv_path", default="external_results.csv")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_results(args.shapecart_source, args.external_csv_path)
    
    if df.empty:
        exit("!!! No data found. Ensure external_results.csv is in the correct folder. !!!")
    
    print(f"--- Models Loaded: {df['model'].unique()} ---")

    metrics = ["train_acc", "val_acc", "test_acc", "elapsed_time", "generalization_gap"]
    existing_metrics = [m for m in metrics if m in df.columns]

    # Grouping and Aggregation
    df_mean = df.groupby(["model", "dataset", "_agg_trial"])[existing_metrics].mean(numeric_only=True).reset_index()
    df_std = df.groupby(["model", "dataset", "_agg_trial"])[existing_metrics].std(numeric_only=True).reset_index()
    
    # Select best hyperparams per dataset
    best_idxs = df_mean.groupby(["dataset", "model"])["val_acc"].idxmax()
    df_best = df_mean.loc[best_idxs].reset_index(drop=True)
    df_best_std = df_std.loc[best_idxs].reset_index(drop=True)
    
    # Save the 5 required files
    outputs = {
        "results_with_hyperparameters.csv": df,
        "best_per_dataset.csv": df_best,
        "best_per_dataset_std.csv": df_best_std,
        "best_avg.csv": df_best.groupby("model")[existing_metrics].mean(numeric_only=True).round(3).reset_index(),
        "best_avg_std.csv": df_best_std.groupby("model")[existing_metrics].mean(numeric_only=True).round(3).reset_index()
    }

    for name, data in outputs.items():
        data.to_csv(os.path.join(args.output_dir, name), index=False)
        
    print(f"✅ Finished. Results are in the '{args.output_dir}' folder.")