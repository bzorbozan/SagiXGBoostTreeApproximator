#!/usr/bin/env python3
"""
Reconstruct hyperparameters for each result JSON using the same deterministic
RNG as xgb_run_new.py / shapefbt_run_new.py (trial_seed = base_seed + trial_id).

Writes one CSV with all JSON fields plus a "hyperparameters" column (JSON string).

Skips anything under results/capability (capability experiments), even when walking from results/.
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


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


def reconstruct_hyperparameters(model: str, base_seed: int, trial_id: int):
    trial_seed = int(base_seed) + int(trial_id)
    rng = np.random.default_rng(trial_seed)
    if model == "XGBoost":
        hp = sample_xgb(rng)
    elif model == "ShapeFBT":
        hp = sample_shapefbt(rng)
    else:
        raise ValueError(f"Unknown model {model!r}; expected 'XGBoost' or 'ShapeFBT'.")
    return hp


# Load samplers from repo (must match experiment scripts)
sample_xgb = _load_sampler_module("xgb_run_new.py")
sample_shapefbt = _load_sampler_module("shapefbt_run_new.py")


def process_json_files(results_dir: Path):
    rows = []
    for path in sorted(results_dir.rglob("*.json")):
        if _is_under_capability_subdir(path, results_dir):
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
            hp = reconstruct_hyperparameters(str(model), base_seed, trial_id)
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
    # Put hyperparameters last; other columns follow sorted union of JSON keys
    extra = ["hyperparameters"]
    key_cols = sorted({k for r in rows for k in r.keys() if k not in extra})
    ordered = key_cols + [c for c in extra if c in df.columns]
    df = df[ordered]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
