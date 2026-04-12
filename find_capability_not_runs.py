import os
import json
import math
import re
from collections import defaultdict

# --- Configuration ---
base_path = [
            './capability_results/shapefbt_capability_results/results/magic', 
            './capability_results/shapefbt_capability_results/results/raisin',
            './capability_results/shapefbt_capability_results/results/bidding',
            ]

def analyze_nan_failures(directory):
    if not os.path.exists(directory):
        print(f"Error: Directory '{directory}' not found.")
        return

    failed_counts = defaultdict(int)
    total_nan_files = 0

    for filename in os.listdir(directory):
        if not filename.endswith(".json"):
            continue
            
        file_path = os.path.join(directory, filename)
        is_nan = False
        
        with open(file_path, 'r') as f:
            try:
                data = json.load(f)
                
                # FIXED: Looking for the correct key "train_acc"
                train_acc = data.get("train_acc") 
                
                # Check for "nan". (Your screenshot shows numbers are saved as strings)
                if isinstance(train_acc, str) and train_acc.strip().lower() == "nan":
                    is_nan = True
                elif isinstance(train_acc, float) and math.isnan(train_acc):
                    is_nan = True
                    
            except json.JSONDecodeError:
                continue

        if is_nan:
            # Extract n and d from the filename
            match = re.search(r'_n(\d+)_d(\d+)_', filename)
            if match:
                n_val = int(match.group(1))
                d_val = int(match.group(2))
                
                # Increment the specific (n, d) counter
                failed_counts[(n_val, d_val)] += 1
                total_nan_files += 1

    # Print the results
    print(f"Total failed folds (NaN train_acc): {total_nan_files}\n")
    print(f"{'n':<5} | {'d':<5} | {'failed_fold_count'}")
    print("-" * 30)
    
    if total_nan_files == 0:
        print("No files with 'nan' train_acc found.")
    else:
        # Sort by n, then by d
        for (n, d) in sorted(failed_counts.keys()):
            count = failed_counts[(n, d)]
            print(f"{n:<5} | {d:<5} | {count}")

# Run the analyzer
for path in base_path:
    print(f"Analyzing {path}")
    analyze_nan_failures(path)
    print("--------------------------------")