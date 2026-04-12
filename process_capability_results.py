import os
import json
import pandas as pd
from pathlib import Path

def process_results():
    input_dir = "capability_results"
    output_dir = "processed_capability_results"
    output_file = os.path.join(output_dir, "capability_results.csv")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []

    # Recursively find all json files
    json_files = Path(input_dir).rglob("*.json")

    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Convert numeric strings to floats
            train_acc = float(data.get("train_acc", 0))
            test_acc = float(data.get("test_acc", 0))
            val_acc = float(data.get("val_acc", 0))
            
            # Generalization Gap = Train - Test
            gen_gap = train_acc - test_acc

            row = {
                "model": data.get("model"),
                "dataset": data.get("dataset"),
                "trial_id": data.get("trial_id"),
                "fold": data.get("fold"),
                "n_estimators": int(data.get("n_estimators", 0)),
                "max_depth": int(data.get("max_depth", 0)),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "test_acc": test_acc,
                "generalization_gap": round(gen_gap, 6),
                "elapsed_time": float(data.get("elapsed_time", 0)),
                "source_file": file_path.name
            }
            all_data.append(row)
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"Skipping {file_path} due to error: {e}")

    if all_data:
        df = pd.DataFrame(all_data)

        # DROP INVALID ENTRIES : remove any row where accuracy metrics are missing or NaN
        initial_count = len(df)
        df = df.dropna(subset=['train_acc', 'test_acc', 'val_acc'])
        final_count = len(df)
        if initial_count > final_count:
            print(f"Filtered out {initial_count - final_count} invalid/empty results.")
        
        # Sorting for a cleaner CSV
        df = df.sort_values(by=["model", "dataset", "trial_id", "fold"])
        
        # Save to the new folder
        df.to_csv(output_file, index=False)
        print(f"Done! Results saved to: {output_file}")

        # Create a summary (Averaging over folds and trials)
        summary_df = df.groupby(["model", "dataset", "n_estimators", "max_depth"]).agg({
            "train_acc": ["mean", "std"],
            "val_acc": ["mean", "std"],
            "test_acc": ["mean", "std"],
            "generalization_gap": ["mean", "std"]
        }).reset_index()

        # Flatten the multi-level columns (e.g., 'test_acc_mean')
        summary_df.columns = [
            '_'.join(col).strip('_') if isinstance(col, tuple) else col 
            for col in summary_df.columns.values
        ]

        summary_file = os.path.join(output_dir, "capability_results_summary.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"Summary saved to: {summary_file}")
    else:
        print(f"No JSON files found in {input_dir}. Did you run the unzip script first?")

if __name__ == "__main__":
    process_results()