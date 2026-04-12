# XGBoostTreeApproximator
This project implements a method that converts a trained XGBoost into a single decision tree.
At this stage, this package is ideal for binary classification challenges.
It can be applied for challenges with a few classes (2-3) but is not recommended for numerous classes, at least for the current implementation.
Please see the included .ipynb file for further explanations of the API.


To run experiments please follow this process:

Generalization experiments:
(make sure all relevant files are in zipped_results)
bash unzip_results.sh
python3 process_results.py
generalization_analysis.ipynb

Optimization capability experiments:
(make sure all relevant files are in zipped_capability_results)
bash unzip_capability_results.sh
python3 process_capability_results.py
capability_analysis.ipynb