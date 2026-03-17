# %%
import xgboost as xg
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
from xgboost import plot_tree
import xgboost as xgb
import matplotlib.pyplot as plt
from FBT import *
from ShapeFBT import *
import traceback

# %% [markdown]
# ## Load Iris dataset

# %%
from datasets import *
train, test, feature_cols, label_col = get_iris_data(1)

# %% [markdown]
# ## Train XGBoost model with default parameters

# %%
model = xg.XGBClassifier()
model.fit(train[feature_cols],train[label_col])

# %% [markdown]
# 

# %%
#    def fit(self,train,feature_cols,label_col, xgb_model, pruned_forest=None, trees_conjunctions_total=None):
#    def __init__(self, outer_tree_max_depth, min_forest_size, max_number_of_conjunctions, pruning_method=None, min_samples_split=10, min_conjunctions_split=2, min_impurity_decrease=0.0, k=2, verbose=False, inner_tree_max_depth=None, inner_tree_criterion='entropy',):

sfbt = ShapeFBT(
    outer_tree_max_depth=5,
    min_forest_size=10,
    max_number_of_conjunctions=1000,
    pruning_method='auc',
    inner_tree_max_depth=5,
    inner_tree_criterion='entropy',
)
sfbt.fit(train, feature_cols, label_col, model)
predictions, probas = sfbt.predict_Xy(test[feature_cols])
print(predictions)
test_acc = (predictions == test[label_col]).mean()
print(f"Test Accuracy: {test_acc:.4f}")

