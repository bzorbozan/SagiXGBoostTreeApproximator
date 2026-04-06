"""
Make sure you run this command in terminal before running this script: tar -xvzf results_name.tar.gz
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

# Only ingest known sweep result JSONs (per-dataset folders under results/).
_RESULT_JSON_PREFIXES = (
    "xgb_results",
    "shapefbt_results",
    "shapecart_results",
    #"rf_results",
    #"sforest_results",
)

# Normalized names for models not listed in farm table.dat / status.txt; use JSON metrics as success signal.
_MODELS_WITHOUT_FARM_SCHEDULE = frozenset({"shapecart"})


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
        "shapecart": "ShapeCART",
        # sforest / rf: keys use script basename unless aliased to match JSON "model"
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


def _has_real_accuracy_metrics(df: pd.DataFrame) -> pd.Series:
    """
    True where train/val/test accuracies are present and finite (not NaN/inf).
    Failed runs often still write JSON with "nan" strings, same idea as excluding non-0 farm status.
    """
    train = pd.to_numeric(df["train_acc"], errors="coerce")
    val = pd.to_numeric(df["val_acc"], errors="coerce")
    test = pd.to_numeric(df["test_acc"], errors="coerce")

    def _finite_series(s):
        return s.notna() & ~s.isin([np.inf, -np.inf])

    return _finite_series(train) & _finite_series(val) & _finite_series(test)


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
            if not any(file.startswith(p) for p in _RESULT_JSON_PREFIXES):
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
    norm_model = df["model"].apply(_normalize_model_name)
    tid = df["trial_id"].fillna(-1).astype(int)
    fld = df["fold"].fillna(-1).astype(int)
    ds = df["dataset"].astype(str)
    in_farm_ok = pd.Series(
        [
            (a, b, c, d) in successful_run_keys
            for a, b, c, d in zip(norm_model, ds, tid, fld)
        ],
        index=df.index,
    )
    # Models not in table.dat have no status 0 key; accept those rows only when metrics look real (like farm-fail -> NaN JSON).
    no_farm_schedule = norm_model.isin(_MODELS_WITHOUT_FARM_SCHEDULE)
    accepted_schedule = no_farm_schedule | in_farm_ok
    n_before_metrics = int(accepted_schedule.sum())
    df_candidate = df.loc[accepted_schedule].copy()
    metrics_ok = _has_real_accuracy_metrics(df_candidate)
    n_dropped_bad_metrics = int((~metrics_ok).sum())
    df = df_candidate.loc[metrics_ok].copy()
    print(
        f"Rows after schedule filter (farm status 0 OR no-farm model list): {n_before_metrics}; "
        f"dropped {n_dropped_bad_metrics} with missing/non-finite train/val/test_acc; "
        f"remaining {len(df)}."
    )
    if df.empty:
        raise ValueError("No rows remain after filtering to successful runs / valid metrics.")

    if "generalization_gap" not in df.columns:
        df["generalization_gap"] = np.abs(df["train_acc"] - df["test_acc"])
    if "n_leaves" not in df.columns:
        df["n_leaves"] = np.nan

    if "max_depth" in df.columns:
        df["max_depth"] = pd.to_numeric(df["max_depth"], errors="coerce")
        # Legacy compatibility: some old ConTree outputs encode depth in trial_id.
        contree_mask = df["model"].astype(str).eq("ConTree")
        df.loc[contree_mask, "trial_id"] = df.loc[contree_mask, "max_depth"]

    # ShapeCART filenames encode outer depth (args.depth); same trial_id can appear at several depths — do not merge those.
    # AGGREGATE TRIAL ID FIX
    is_shapecart = df["model"].astype(str).eq("ShapeCART")
    df["_agg_trial"] = df["trial_id"].astype(str)
    if "max_depth" in df.columns and is_shapecart.any():
        md = df["max_depth"]
        both = is_shapecart & md.notna()
        df.loc[both, "_agg_trial"] = (
            df.loc[both, "trial_id"].astype(int).astype(str)
            + "_"
            + md.loc[both].astype(int).astype(str)
        )

    models = sorted(df["model"].dropna().astype(str).unique().tolist())
    datasets = sorted(df["dataset"].dropna().astype(str).unique().tolist())
    numeric_cols = ["train_acc", "val_acc", "test_acc", "n_leaves", "generalization_gap", "elapsed_time"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    # AGGREGATE TRIAL ID FIX
    df_mean = df.groupby(["model", "dataset", "_agg_trial"])[numeric_cols].mean().reset_index()
    df_std = df.groupby(["model", "dataset", "_agg_trial"])[numeric_cols].std().reset_index()

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
