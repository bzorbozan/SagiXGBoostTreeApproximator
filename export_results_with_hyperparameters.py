#!/usr/bin/env python3
"""
Reconstruct hyperparameters for each result JSON.

- XGBoost / ShapeFBT: same deterministic RNG as xgb_run_new.py / shapefbt_run_new.py
  (trial_seed = base_seed + trial_id).
- ShapeCART: RNG replay via sample_shapecart_hyperparameters (same search space as Optuna);
  max_depth is taken from the JSON (CLI args.depth), not from the sampler.

Writes one CSV with all JSON fields plus a "hyperparameters" column (JSON string).

Skips anything under results/capability (capability experiments), even when walking from results/.
Only ingests JSON filenames matching the same prefixes as process_results.py.
"""

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent

# Keep in sync with process_results.py
_RESULT_JSON_PREFIXES = (
    "xgb_results",
    "shapefbt_results",
    "shapecart_results",
    #"rf_results",
    #"sforest_results",
)

# Normalized names for models not listed in farm table.dat / status.txt; use JSON metrics as success signal.
_MODELS_WITHOUT_FARM_SCHEDULE = frozenset({"shapecart"})

# JSON keys treated as run metadata / metrics (not copied into hyperparameters for non-RNG models)
_JSON_NON_HP_KEYS = {
    "model",
    "trial_id",
    "fold",
    "dataset",
    "random_seed",
    "base_seed",
    "train_acc",
    "val_acc",
    "test_acc",
    "elapsed_time",
    "generalization_gap",
    "actual_depth",
    "n_leaves",
    "failed",
}

_sample_xgb = None
_sample_shapefbt = None


def _normalize_model_name(name):
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _load_successful_run_keys(table_path: Path, status_path: Path):
    """
    Build successful run keys from farm scheduler files.
    Returns set of tuples: (normalized_model, dataset, trial_id, fold)
    """
    status_map = {}
    with open(status_path, "r", encoding="utf-8") as f:
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
    }

    keys = set()
    with open(table_path, "r", encoding="utf-8") as f:
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
    train = pd.to_numeric(df["train_acc"], errors="coerce")
    val = pd.to_numeric(df["val_acc"], errors="coerce")
    test = pd.to_numeric(df["test_acc"], errors="coerce")

    def _finite_series(s):
        return s.notna() & ~s.isin([np.inf, -np.inf])

    return _finite_series(train) & _finite_series(val) & _finite_series(test)


def _apply_consistent_success_filter(df: pd.DataFrame, table_path: Path, status_path: Path) -> pd.DataFrame:
    """Match process_results.py success filtering behavior."""
    required = ["model", "dataset", "trial_id", "fold", "train_acc", "val_acc", "test_acc"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for success filtering: {missing}")

    successful_run_keys = _load_successful_run_keys(table_path, status_path)
    if not successful_run_keys:
        raise ValueError("No successful runs found from status file. Check --table_path and --status_path.")

    norm_model = df["model"].apply(_normalize_model_name)
    tid = pd.to_numeric(df["trial_id"], errors="coerce").fillna(-1).astype(int)
    fld = pd.to_numeric(df["fold"], errors="coerce").fillna(-1).astype(int)
    ds = df["dataset"].astype(str)

    in_farm_ok = pd.Series(
        [(a, b, c, d) in successful_run_keys for a, b, c, d in zip(norm_model, ds, tid, fld)],
        index=df.index,
    )
    no_farm_schedule = norm_model.isin(_MODELS_WITHOUT_FARM_SCHEDULE)
    accepted_schedule = no_farm_schedule | in_farm_ok

    df_candidate = df.loc[accepted_schedule].copy()
    metrics_ok = _has_real_accuracy_metrics(df_candidate)
    filtered = df_candidate.loc[metrics_ok].copy()
    return filtered

# AGGREGATE TRIAL ID FIX
def _add_aggregated_trial_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep a process_results.py-consistent aggregation key.
    For ShapeCART, aggregate key is trial_id + max_depth.
    For other models, it is just trial_id.
    """
    out = df.copy()
    out["trial_id_agg"] = out["trial_id"].astype(str)
    if "max_depth" not in out.columns:
        return out

    is_shapecart = out["model"].astype(str).eq("ShapeCART")
    trial_num = pd.to_numeric(out["trial_id"], errors="coerce")
    max_depth_num = pd.to_numeric(out["max_depth"], errors="coerce")
    both = is_shapecart & trial_num.notna() & max_depth_num.notna()
    out.loc[both, "trial_id_agg"] = (
        trial_num.loc[both].astype(int).astype(str)
        + "_"
        + max_depth_num.loc[both].astype(int).astype(str)
    )
    return out


def _ensure_import_paths_for_run_scripts():
    """
    `importlib` loading of tests/xgb_run_new.py does not put `tests/` on sys.path,
    so `from data_utils import *` fails unless we mirror a normal run (tests + repo root
    for ShapeFBT, etc.).
    """
    root = str(ROOT)
    tests = str(ROOT / "tests")
    if root not in sys.path:
        sys.path.insert(0, root)
    if tests not in sys.path:
        sys.path.insert(0, tests)


def _is_under_capability_subdir(path: Path, results_dir: Path) -> bool:
    """True if path is inside a directory named 'capability' under results_dir."""
    try:
        rel = path.resolve().relative_to(results_dir.resolve())
    except ValueError:
        return False
    return "capability" in rel.parts


def _load_sampler_module(script_name: str):
    _ensure_import_paths_for_run_scripts()
    path = ROOT / "tests" / script_name
    spec = importlib.util.spec_from_file_location(script_name.replace(".py", ""), path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.sample_hyperparameters


def _json_safe(value):
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def _hyperparams_json_string(hp: dict) -> str:
    return json.dumps(_json_safe(hp), separators=(",", ":"), sort_keys=True)


def sample_shapecart_hyperparameters(rng):
    """
    Sample hyperparameters randomly matching the previous Optuna search space.

    Must stay in sync with the ShapeCART runner (shapecart_run_new.py or equivalent).
    """
    criterion = rng.choice(["gini", "entropy"])

    min_samples_split_exp = rng.integers(1, 6)
    min_samples_split = 2 ** int(min_samples_split_exp)

    min_samples_leaf = rng.integers(1, 33)

    min_impurity_decrease = rng.choice([0.0, 1e-4, 5e-4, 1e-3, 5e-3, 0.01])

    choices = [x * 4 for x in range(1, 17)]
    inner_max_leaf_nodes = int(rng.choice(choices))

    inner_min_samples_leaf = rng.choice([1, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])

    return {
        "criterion": str(criterion),
        "min_samples_split": int(min_samples_split),
        "min_samples_leaf": int(min_samples_leaf),
        "min_impurity_decrease": float(min_impurity_decrease),
        "inner_max_leaf_nodes": int(inner_max_leaf_nodes),
        "inner_min_samples_leaf": inner_min_samples_leaf,
    }


def _get_sample_xgb():
    global _sample_xgb
    if _sample_xgb is None:
        _sample_xgb = _load_sampler_module("xgb_run_new.py")
    return _sample_xgb


def _get_sample_shapefbt():
    global _sample_shapefbt
    if _sample_shapefbt is None:
        _sample_shapefbt = _load_sampler_module("shapefbt_run_new.py")
    return _sample_shapefbt


def _coerce_scalar(v):
    if isinstance(v, str):
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v, 10)
        except ValueError:
            return v
    return v


def _hyperparams_from_result_json(data: dict) -> dict:
    """Use logged fields in the JSON when there is no RNG sampler module (e.g. ShapeCART)."""
    hp = {}
    for k, v in data.items():
        if k in _JSON_NON_HP_KEYS:
            continue
        hp[k] = _coerce_scalar(v)
    return hp


def reconstruct_hyperparameters(model: str, base_seed: int, trial_id: int, data: dict) -> dict:
    trial_seed = int(base_seed) + int(trial_id)
    rng = np.random.default_rng(trial_seed)
    if model == "XGBoost":
        return _get_sample_xgb()(rng)
    if model == "ShapeFBT":
        return _get_sample_shapefbt()(rng)
    if model == "ShapeCART":
        hp = sample_shapecart_hyperparameters(rng)
        md = data.get("max_depth")
        if md is not None and str(md).strip() != "":
            hp["max_depth"] = _coerce_scalar(md)
        return hp
    return _hyperparams_from_result_json(data)


def process_json_files(results_dir: Path):
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        if _is_under_capability_subdir(path, results_dir):
            continue
        if not any(path.name.startswith(p) for p in _RESULT_JSON_PREFIXES):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"Skipping {path}: {e}", file=sys.stderr)
            continue

        if not isinstance(data, dict):
            print(f"Skipping {path}: top-level JSON is not an object", file=sys.stderr)
            continue

        model = data.get("model")
        try:
            base_seed = int(data["base_seed"])
            trial_id = int(data["trial_id"])
        except (KeyError, TypeError, ValueError) as e:
            print(f"Skipping {path}: missing base_seed/trial_id ({e})", file=sys.stderr)
            continue

        try:
            hp = reconstruct_hyperparameters(str(model), base_seed, trial_id, data)
        except Exception as e:
            print(f"Skipping {path}: hyperparameter reconstruction failed ({e})", file=sys.stderr)
            continue

        row = dict(data)
        row["hyperparameters"] = _hyperparams_json_string(hp)
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "results",
        help="Directory containing per-dataset JSON results (default: ./results)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results_with_hyperparameters.csv",
        help="Output CSV path (default: ./results_with_hyperparameters.csv)",
    )
    parser.add_argument("--table_path", type=Path, default=ROOT / "farm1/table.dat", help="Path to farm table.dat")
    parser.add_argument("--status_path", type=Path, default=ROOT / "farm1/status.txt", help="Path to farm status.txt")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        print(f"Not a directory: {results_dir}", file=sys.stderr)
        sys.exit(1)

    rows = process_json_files(results_dir)
    if not rows:
        print("No rows written.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = _apply_consistent_success_filter(df, args.table_path, args.status_path)
    df = _add_aggregated_trial_id(df)
    if df.empty:
        print("No rows remain after filtering to successful runs / valid metrics.", file=sys.stderr)
        sys.exit(1)
    # Put hyperparameters last; other columns follow sorted union of JSON keys
    extra = ["hyperparameters"]
    key_cols = sorted({k for k in df.columns if k not in extra})
    ordered = key_cols + [c for c in extra if c in df.columns]
    df = df[ordered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
