import argparse
import json
import os
import re

import numpy as np
import pandas as pd


def _normalize_model_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _load_successful_run_keys(table_path, status_path):
    """
    Build successful run keys from farm scheduler files.
    Returns set of tuples: (normalized_model, dataset, trial_id, fold)
    """
    status_map = {}
    with open(status_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            run_id = int(parts[0])
            status_map[run_id] = parts[1]

    model_alias = {
        "xgb": "XGBoost",
        "shapefbt": "ShapeFBT",
    }

    keys = set()
    with open(table_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            run_id = int(parts[0])
            status = status_map.get(run_id)
            if status != "0":
                continue

            dataset_match = re.search(r"--dataset\s+([^\s]+)", line)
            trial_match = re.search(r"--trial-id\s+([^\s]+)", line)
            fold_match = re.search(r"--fold\s+([^\s]+)", line)
            script_match = re.search(r"/tests/([^/\s]+)_run_new\.py", line)

            if not (dataset_match and trial_match and fold_match and script_match):
                continue

            script_prefix = script_match.group(1)
            model_name = model_alias.get(script_prefix, script_prefix)
            dataset = dataset_match.group(1)
            trial_id = int(float(trial_match.group(1)))
            fold = int(float(fold_match.group(1)))
            keys.add((_normalize_model_name(model_name), dataset, trial_id, fold))

    return keys


def load_all_results(results_dir, dataset=None, delete_json=False):
    """Load all result JSON files and optionally merge with existing results.csv."""
    existing_csv = os.path.join(results_dir, "results.csv")
    df_existing = pd.read_csv(existing_csv) if os.path.exists(existing_csv) else None

    walk_dir = os.path.join(results_dir, dataset) if dataset else results_dir
    print(f"Walking through {walk_dir} to find result JSON files...")

    results = []
    for root, _, files in os.walk(walk_dir):
        for file in files:
            if not file.endswith(".json"):
                continue
            path = os.path.join(root, file)
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                results.append(data)
                if delete_json:
                    os.remove(path)
            except (json.JSONDecodeError, OSError) as exc:
                print(f"Skipping unreadable file {path}: {exc}")

    print(f"Loaded {len(results)} result files.")
    if not results and df_existing is None:
        return pd.DataFrame()

    df_new = pd.DataFrame(results) if results else pd.DataFrame()

    # Normalize expected numeric fields from string JSON values.
    float_cols = [
        "train_acc",
        "val_acc",
        "test_acc",
        "elapsed_time",
        "actual_depth",
        "n_leaves",
        "generalization_gap",
    ]
    int_cols = ["trial_id", "fold", "max_depth", "random_seed", "base_seed", "failed"]

    for col in float_cols:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype("float")
    for col in int_cols:
        if col in df_new.columns:
            df_new[col] = pd.to_numeric(df_new[col], errors="coerce").astype("Int64")

    # Merge with existing aggregate if present.
    if df_existing is not None and not df_new.empty:
        df = pd.concat([df_existing, df_new], ignore_index=True)
    elif df_existing is not None:
        df = df_existing
    else:
        df = df_new

    # Deduplicate by stable identifiers that exist in the data.
    dedupe_cols = [
        "model",
        "dataset",
        "trial_id",
        "fold",
        "max_depth",
        "random_seed",
        "base_seed",
    ]
    dedupe_cols = [c for c in dedupe_cols if c in df.columns]
    if dedupe_cols:
        df = df.drop_duplicates(subset=dedupe_cols).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    df.to_csv(existing_csv, index=False)
    print(f"Wrote merged table to {existing_csv} ({len(df)} rows).")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results", help="Directory with per-dataset JSON results")
    parser.add_argument("--dataset", default=None, help="Optional dataset subfolder to process")
    parser.add_argument("--output_dir", default="processed_results", help="Directory to save summary CSVs")
    parser.add_argument("--delete_json", action="store_true", help="Delete JSON files after reading")
    parser.add_argument("--table_path", default="farm1/table.dat", help="Path to farm table.dat")
    parser.add_argument("--status_path", default="farm1/status.txt", help="Path to farm status.txt")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_all_results(args.dir, dataset=args.dataset, delete_json=args.delete_json)

    if df.empty:
        print("No results found; nothing to aggregate.")
        raise SystemExit(0)

    required = ["model", "dataset", "trial_id", "train_acc", "val_acc", "test_acc", "elapsed_time"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in result JSONs: {missing}")

    if "fold" not in df.columns:
        raise ValueError("Missing required column 'fold' in result JSONs for status filtering.")
    # CHANGE FROM ORIGINAL - WE ONLY CARE ABOUT SUCCESSFUL RUNS!!
    successful_run_keys = _load_successful_run_keys(args.table_path, args.status_path)
    if not successful_run_keys:
        raise ValueError("No successful runs found from status file. Check --table_path and --status_path.")
    df["trial_id"] = pd.to_numeric(df["trial_id"], errors="coerce")
    df["fold"] = pd.to_numeric(df["fold"], errors="coerce")
    # status.txt shows  0 = successful. 127 = timeout (dw about these), 137 = OOM for each run in table.dat. Here, we will only consider successful runs.
    df = df[
        df.apply(
            lambda row: (
                _normalize_model_name(row["model"]),
                str(row["dataset"]),
                int(row["trial_id"]) if pd.notna(row["trial_id"]) else -1,
                int(row["fold"]) if pd.notna(row["fold"]) else -1,
            )
            in successful_run_keys,
            axis=1,
        )
    ].copy()
    print(f"Rows after status-based success filter: {len(df)}")
    if df.empty:
        raise ValueError("No rows remain after filtering to successful runs.")

    if "generalization_gap" not in df.columns:
        df["generalization_gap"] = np.abs(df["train_acc"] - df["test_acc"])
    if "n_leaves" not in df.columns:
        df["n_leaves"] = np.nan

    if "max_depth" in df.columns:
        df["max_depth"] = pd.to_numeric(df["max_depth"], errors="coerce")
        # Legacy compatibility: some old ConTree outputs encode depth in trial_id.
        contree_mask = df["model"].astype(str).eq("ConTree")
        df.loc[contree_mask, "trial_id"] = df.loc[contree_mask, "max_depth"]

    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())
    numeric_cols = ["train_acc", "val_acc", "test_acc", "n_leaves", "generalization_gap", "elapsed_time"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    df_mean = df.groupby(["model", "dataset", "trial_id"])[numeric_cols].mean().reset_index()
    df_std = df.groupby(["model", "dataset", "trial_id"])[numeric_cols].std().reset_index()

    best_idxs = df_mean.groupby(["dataset", "model"])["val_acc"].idxmax().dropna().astype(int)
    df_best = df_mean.loc[best_idxs].reset_index(drop=True)
    df_best_std = df_std.loc[best_idxs].reset_index(drop=True)

    multi_idx = pd.MultiIndex.from_product([datasets, models], names=["dataset", "model"])
    keep_cols = ["dataset", "model", "train_acc", "test_acc", "n_leaves", "generalization_gap", "elapsed_time"]
    keep_cols = [c for c in keep_cols if c in df_best.columns]

    df_best = df_best[keep_cols].set_index(["dataset", "model"]).reindex(multi_idx).round(3)
    df_best_std = df_best_std[keep_cols].set_index(["dataset", "model"]).reindex(multi_idx).round(3)

    best_path = os.path.join(args.output_dir, "best_per_dataset.csv")
    best_std_path = os.path.join(args.output_dir, "best_per_dataset_std.csv")
    df_best.to_csv(best_path)
    df_best_std.to_csv(best_std_path)

    df_best_avg = df_best.groupby("model").mean().round(3).reset_index()
    df_best_avg_std = df_best_std.groupby("model").mean().round(3).reset_index()
    avg_path = os.path.join(args.output_dir, "best_avg.csv")
    avg_std_path = os.path.join(args.output_dir, "best_avg_std.csv")
    df_best_avg.to_csv(avg_path, index=False)
    df_best_avg_std.to_csv(avg_std_path, index=False)

    print(f"Wrote: {best_path}")
    print(f"Wrote: {best_std_path}")
    print(f"Wrote: {avg_path}")
    print(f"Wrote: {avg_std_path}")
