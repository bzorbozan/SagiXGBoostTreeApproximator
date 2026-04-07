"""
Aggregate experiment JSON results into summary CSVs from specific subdirectory structures.

This script processes raw experiment outputs stored in a nested directory format, 
specifically designed to handle multiple model families without overlapping results.

Key Features & Logic:
1. Directory-to-Prefix Mapping:
   - Only scans specific folders under 'all_results/' for designated file prefixes:
     * 'all_results/xgb_shapefbt_results' -> [xgb_results, shapefbt_results]
     * 'all_results/sagi_results'         -> [fbt_results]
     * 'all_results/shapecart_results'    -> [shapecart_results]

2. Success Filtering (Metric-Based):
   - Does NOT use 'status.txt' or external farm tables. 
   - A run is considered successful only if 'train_acc', 'val_acc', and 'test_acc' 
     contain valid numeric values; files containing "nan" are filtered out.

3. Duplicate Identification (Reporting Only):
   - Scans for duplicates based on [dataset, model, trial_id, max_depth, fold].
   - Performs "Exact Payload" deduplication by comparing the full JSON content.
   - IMPORTANT: This script writes duplicate details to 'all_results_duplicates.csv' 
     for manual review; it does not delete or modify the source JSON files.

4. Model-Specific Handling:
   - ShapeCART: Uses an aggregated key (trial_id + max_depth) to ensure unique 
     identification since trial_ids are shared across different depths.
   - Generalization Gap: Automatically calculates |train_acc - test_acc|.

5. Fresh Execution:
   - Does NOT merge with existing 'results.csv' files. 
   - Every execution performs a clean scan and generates new output files from scratch.

6. Winner Selection & Statistical Aggregation:
   - Per Dataset: Chooses the "winner" for every (dataset, model) pair based on 
     the highest validation accuracy (val_acc).
   - Means & Std Devs: Calculates both the average performance and the standard 
     deviation for individual datasets and global model averages.

Outputs (saved to /processed_results):
- results.csv: A complete flat table of all valid, non-filtered experiment runs.
- best_per_dataset.csv / best_per_dataset_std.csv: Winning trial metrics (Mean & SD).
- best_avg.csv / best_avg_std.csv: Global average performance (Mean & SD).
- all_results_duplicates.csv: A list of flagged duplicate files (if any exist).
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

def check_duplicates(df_new, output_dir):
    """
    Identifies duplicates and writes them to all_results_duplicates.csv.
    """
    duplicate_log = []
    
    id_cols = ['dataset', 'model', 'trial_id', 'max_depth', 'fold']
    existing_id_cols = [c for c in id_cols if c in df_new.columns]
    
    id_dupes = df_new[df_new.duplicated(subset=existing_id_cols, keep=False)]
    
    for _, row in id_dupes.iterrows():
        duplicate_log.append({
            "type": "ID DUPE",
            "model": row.get('model'),
            "dataset": row.get('dataset'),
            "trial_id": row.get('trial_id'),
            "max_depth": row.get('max_depth'),
            "fold": row.get('fold'),
            "file_path": row.get('file_path')
        })

    if 'fingerprint' in df_new.columns:
        content_dupes = df_new[df_new.duplicated(subset=['fingerprint'], keep=False)]
        for _, row in content_dupes.iterrows():
            duplicate_log.append({
                "type": "CONTENT DUPE",
                "model": row.get('model'),
                "dataset": row.get('dataset'),
                "trial_id": row.get('trial_id'),
                "max_depth": row.get('max_depth'),
                "fold": row.get('fold'),
                "file_path": row.get('file_path')
            })

    if duplicate_log:
        dupe_df = pd.DataFrame(duplicate_log).drop_duplicates()
        dupe_path = os.path.join(output_dir, "all_results_duplicates.csv")
        dupe_df.to_csv(dupe_path, index=False)
        print(f"⚠️  Found {len(dupe_df)} duplicate entries. Details saved to: {dupe_path}")
    else:
        print("✅ No duplicates detected.")

def load_results():
    results = []
    for folder, prefixes in FOLDER_MAP.items():
        if not os.path.exists(folder):
            continue
        print(f"Searching in {folder}...")
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".json") and any(file.startswith(p) for p in prefixes):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "r") as f:
                            data = json.load(f)
                        data['file_path'] = path
                        data['fingerprint'] = json.dumps(data, sort_keys=True)
                        results.append(data)
                    except Exception as e:
                        print(f"Could not read {path}: {e}")

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    numeric_cols = ["train_acc", "val_acc", "test_acc", "elapsed_time", "max_depth", "trial_id", "fold"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    initial_count = len(df)
    df = df.dropna(subset=["train_acc", "val_acc", "test_acc"])
    print(f"Loaded {initial_count} files; {len(df)} kept after filtering 'nan' accuracies.")
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Now defaults back to "processed_results" folder
    parser.add_argument("--output_dir", default="processed_results", help="Directory to save CSVs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    df = load_results()
    if df.empty:
        print("No valid results found. Exiting.")
        exit()

    check_duplicates(df, args.output_dir)

    # ShapeCART Aggregation
    is_shapecart = df["model"].astype(str).apply(_normalize_model_name).str.contains("shapecart")
    df["_agg_trial"] = df["trial_id"].astype(str)
    if "max_depth" in df.columns:
        df.loc[is_shapecart, "_agg_trial"] = (
            df.loc[is_shapecart, "trial_id"].fillna(0).astype(int).astype(str) + "_" + 
            df.loc[is_shapecart, "max_depth"].fillna(0).astype(int).astype(str)
        )

    if "generalization_gap" not in df.columns:
        df["generalization_gap"] = np.abs(df["train_acc"] - df["test_acc"])

    metric_cols = ["train_acc", "val_acc", "test_acc", "elapsed_time", "generalization_gap"]
    existing_metrics = [m for m in metric_cols if m in df.columns]
    
    # 1. Calculate Means and Standard Deviations
    df_mean = df.groupby(["model", "dataset", "_agg_trial"])[existing_metrics].mean().reset_index()
    df_std = df.groupby(["model", "dataset", "_agg_trial"])[existing_metrics].std().reset_index()
    
    # 2. Identify Winners (based on mean val_acc)
    best_idxs = df_mean.groupby(["dataset", "model"])["val_acc"].idxmax()
    
    # 3. Create Per-Dataset Tables (Mean and SD)
    df_best = df_mean.loc[best_idxs].reset_index(drop=True)
    df_best_std = df_std.loc[best_idxs].reset_index(drop=True)

    # 4. Create Global Average Tables (Mean and SD)
    df_best_avg = df_best.groupby("model")[existing_metrics].mean().round(3).reset_index()
    df_best_avg_std = df_best_std.groupby("model")[existing_metrics].mean().round(3).reset_index()

    # 5. Save all 5 resulting CSVs to /processed_results
    paths = {
        "results.csv": df,
        "best_per_dataset.csv": df_best,
        "best_per_dataset_std.csv": df_best_std,
        "best_avg.csv": df_best_avg,
        "best_avg_std.csv": df_best_avg_std
    }

    print(f"\n✅ Files created in: {os.path.abspath(args.output_dir)}")
    for name, data in paths.items():
        full_path = os.path.join(args.output_dir, name)
        data.to_csv(full_path, index=False)
        print(f"   - {name}")