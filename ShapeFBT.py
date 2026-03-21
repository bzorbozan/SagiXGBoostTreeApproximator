""""
This module contains the ShapeFBT which acts as the outer tree in our Sagi-ShapeCART Algorithm
"""
from conjunctionset import *
from tree import *
from pruning import *
from FBT import *
from TreesExtraction import extractConjunctionSetsFromForest
from pruning import Pruner

from utils import softmax
import heapq
import numpy as np
import pandas as pd
import traceback

# Helper Functions

def _leaf_proba(conjunctions):
    """Mean softmax over conjunctions -- mirrors Tree.predict_instance_proba."""
    if not conjunctions:
        raise ValueError('_leaf_proba called with an empty conjunction list.')
    return np.array([softmax(c.label_probas) for c in conjunctions]).mean(axis=0)[0]

def _route_through_buckets(fbt, X_node, feat_idx):
    """
    Route each row of X_node through fbt.tree and return the
    bucket_assignment (0..k-1) stored on the reached leaf by map_to_buckets.
    fbt.tree was built on a single feature, so feat_idx selects that column.
    """
    branching = np.zeros(len(X_node), dtype=np.int32)
    for i, inst in enumerate(X_node):
        node = fbt.tree
        while node.selected_feature is not None:
            if inst[feat_idx] >= node.selected_value:
                node = node.left
            else:
                node = node.right
        branching[i] = node.bucket_assignment
    return branching

# #ShapeFBT Class

class ShapeFBT():
    def __init__(self, 
                 outer_tree_max_depth, 
                 min_forest_size, 
                 max_number_of_conjunctions, 
                 pruning_method=None, 
                 min_samples_split=10, 
                 min_conjunctions_split=2, 
                 min_impurity_decrease=0.0, 
                 k=2, 
                 verbose=False,
                 inner_tree_max_depth=None,
                 inner_tree_criterion='entropy',
                 ):

        self.verbose = verbose

        # Simple helper for conditional logging
        # (mirrors verbose_print usage in ShapeCART-style code).
        def verbose_print_fn(*args, **kwargs):
            if self.verbose:
                print(*args, **kwargs)
        self.verbose_print = verbose_print_fn

        # Attributes needed to calculate conjunction sets
        self.pruning_method = pruning_method
        self.min_forest_size = min_forest_size
        self.max_number_of_conjunctions = max_number_of_conjunctions
        self.pruning_method = pruning_method

        # Outer tree depth (ShapeFBT itself)
        self.outer_tree_max_depth = outer_tree_max_depth

        # Inner tree depth (FBT trees used at each node).
        # If not provided, default to the same depth as the outer tree.
        if inner_tree_max_depth is None:
            inner_tree_max_depth = outer_tree_max_depth
        self.inner_tree_max_depth = inner_tree_max_depth
        self.inner_tree_criterion = inner_tree_criterion

        # Related to inner loop
        self.min_impurity_decrease = min_impurity_decrease
        self.k = k
        self.min_samples_split      = min_samples_split
        self.min_conjunctions_split = min_conjunctions_split

        # Will be populated when fitting, if a feature_dict is provided
        self.index_dict = None
        self.cat_dict = None

    def configure_feature_dict(self, X, feature_dict):
        """
        Configure feature grouping based on a feature_dict.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        feature_dict : dict or None
            If not None, expected format:
                {key: [list_of_column_indices]}
            where each list groups columns that belong to one (possibly
            categorical) logical feature.

        Returns
        -------
        index_dict : dict[int, list[int]]
            Maps logical feature id to list of column indices.
        cat_dict : dict[int, bool]
            True if the logical feature is categorical (len(indices) > 1).
        """
        if feature_dict is None:
            self.verbose_print('No feature dict provided, assuming all features are continuous')
            index_dict = {i: [i] for i in range(X.shape[1])}
            cat_dict = {i: False for i in range(X.shape[1])}
            return index_dict, cat_dict
        else:
            # Copy and validate that indices are unique
            index_dict = feature_dict.copy()
            all_cat_idxs = []
            for val in index_dict.values():
                all_cat_idxs.extend(val)

            assert len(all_cat_idxs) == len(set(all_cat_idxs)), 'Feature indices must be unique'
            for i in range(X.shape[1]):  # fill in gaps, assume non-categorical
                if i not in all_cat_idxs:
                    index_dict[i] = [i]
                    self.verbose_print(f'Adding index {i} as continuous')

            cat_dict = {}
            for k, v in index_dict.items():
                cat_dict[k] = len(v) > 1

            cat_list = [k for k, v in cat_dict.items() if v]
            cont_list = [k for k, v in cat_dict.items() if not v]
            self.verbose_print(f"Categorical features: {cat_list}, Continuous features: {cont_list}")
            return index_dict, cat_dict

    def _fit_one_feature(self, conjunctions, feat_col, feat_idx, label_col):
        """
        Fit FBT on a single feature at the current node, then map_to_buckets.

        Uses the splitting points of one feature only, forcing Tree.split() to only iterate thorugh that feature's values

        Returns (fbt, result) or (None, None) if no valid split.
        """
        fbt = FBT(
            inner_tree_max_depth=self.inner_tree_max_depth,
            min_forest_size=self.min_forest_size,
            max_number_of_conjunctions=self.max_number_of_conjunctions,
            pruning_method=None,
            criterion=self.inner_tree_criterion,
        )
        splitting_points_one_feature =  {feat_idx: self.cs.splitting_points[feat_idx]} if feat_idx in self.cs.splitting_points else {}
        # print("the split points are:", splitting_points_one_feature)
        # print(self.cs.splitting_points)
        fbt.fit( 
            conj_set=conjunctions,
            splitting_points=splitting_points_one_feature,
            feature_cols=[feat_col],
            label_col=label_col,
        )
        n_leaves = fbt.n_leaves
        if n_leaves < 2:
            return None, None, None, None
        
        result, left_conjunctions, right_conjunctions = fbt.map_to_buckets(k=self.k)

        if result['impurity_decrease'] <= self.min_impurity_decrease:
            return None, None, None, None

        return fbt, result, left_conjunctions, right_conjunctions

    def _best_feature_split(self, 
                            conjunctions, 
                            feature_cols, label_col):
        """
        Call FBT.fit() + map_to_buckets() for every feature on the node's
        data. Return the feature with the highest impurity decrease.

        Returns (feat_col, feat_idx, fbt, result)
        or      (None, None, None, None) if no valid split.
        """
        best_imp  = self.min_impurity_decrease
        best      = (None, None, None, None, None)

        # Iterate over *columns* as separate features, regardless of feature_dict.
        # This preserves the original scalar splitting logic.
        for feat_idx, feat_col in enumerate(feature_cols):
            # print("Now trying to split for feat:", feat_col)
            
            fbt, result, left_con, right_cons = self._fit_one_feature(
                conjunctions=conjunctions, 
                feat_col=feat_col, 
                feat_idx=feat_idx, 
                label_col=label_col,
            )

            if fbt is not None and result['impurity_decrease'] > best_imp:
                best_imp = result['impurity_decrease']
                best     = (feat_col, feat_idx, fbt, result, [left_con, right_cons])
        return best

    # def _select_best_feature():
    def fit(self,
            train,
            feature_cols,
            label_col,
            xgb_model,
            pruned_forest=None,
            trees_conjunctions_total=None,
            feature_dict=None,
            ):
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.int_cols = [k for k,v in train[feature_cols].dtypes.items() if 'int' in str(v)]
        self.xgb_model = xgb_model

        # Configure feature grouping / categorical handling, if requested
        X_matrix = train[feature_cols].values
        self.index_dict, self.cat_dict = self.configure_feature_dict(X_matrix, feature_dict)

        if pruned_forest is None or trees_conjunctions_total is None:
            self.trees_conjunctions_total = extractConjunctionSetsFromForest(self.xgb_model,train[self.label_col].unique(),self.feature_cols)
            print('Start pruning')
            self.prune(train)
        else:
            self.pruner = Pruner()
            self.trees_conjunctions_total = trees_conjunctions_total
            self.trees_conjunctions = pruned_forest
        self.cs = ConjunctionSet(max_number_of_conjunctions=self.max_number_of_conjunctions)
        self.cs.fit(self.trees_conjunctions,train, feature_cols,label_col,int_features=self.int_cols)

        # Now we pass this into the FBT without needing to re-calculate it during every loop
        # we will have splitting_points only be called for a single feature (type still has to be a dict)
        # honestly you can just call the rest directly, it is just 2 lines. but still keep there for modularity and to not mess with it as much 
        # then you call map to buckets which give you the L/R results -> 0/1 

        # ── Step 2: best-first TDIDT loop ────────────────────────────────
        # print('ShapeFBT: starting TDIDT loop ... (pls workk)')
        #X = train[feature_cols].values
        #y = train[label_col].values

        # initialise root (node 0)
        self.shape_nodes        = [None]
        self.shape_conjunctions = [self.cs.conjunctions] # conjunctinos at that node
        self.shape_children     = [[]] # you look up shape_children[node_idx][branch] to see where to go next
        self.shape_depths       = [0] # do we care to keep track of this
        self.shape_is_leaf      = [True]
        self.shape_n_leaves     = 1 # not needed

        feat_col, feat_idx, fbt, result, conjunction_split = self._best_feature_split(self.cs.conjunctions, feature_cols, label_col)
        if feat_col is None:
            print('ShapeFBT: no valid split at root -- single-leaf tree.')
            return self
        # print("---------------------------STARTING ACTUAL LOOP ---------------")
        heap = []
        heapq.heappush(heap, (-result['impurity_decrease'], 0, feat_col, feat_idx, fbt, result, conjunction_split))

        while heap:
            neg_imp, node_idx, feat_col, feat_idx, fbt, result, conjunction_split = heapq.heappop(heap)

            if self.outer_tree_max_depth is not None and self.shape_depths[node_idx] >= self.outer_tree_max_depth:
                continue

            node_conjunctions = self.shape_conjunctions[node_idx]

            if len(node_conjunctions) < self.min_conjunctions_split:
                continue

            # commit as internal node
            self.shape_nodes[node_idx]    = (feat_col, feat_idx, fbt)
            self.shape_is_leaf[node_idx]  = False
            self.shape_children[node_idx] = []

            # branching = _route_through_buckets(fbt, X_node, feat_idx)
            # k_actual  = int(branching.max()) + 1 # in this case k actual would be the len(partitions)
            # np.zeros(len(X_node), dtype=np.int32) -> each entry is either 0 or 1 (direction for conjunction)
            # partitions = _partition_conjunctions(node_conjunctions, X_node, branching, k_actual) # dimensionality here: [[] for _ in range(k)]
            
            partitions = conjunction_split

            for direction in range(len(partitions)): # 0 for left, 1 for right
                child_conjunctions = partitions[direction]
                child_idx = len(self.shape_nodes) # this child is the iˆth node to be added to the heap
                child_depth = self.shape_depths[node_idx] + 1
                
                # append placeholders so child_idx is valid in all parallel lists
                self.shape_nodes.append(None)
                self.shape_conjunctions.append(child_conjunctions)
                self.shape_children.append([])
                self.shape_depths.append(child_depth)
                self.shape_is_leaf.append(True)
                self.shape_children[node_idx].append(child_idx)

                if self.outer_tree_max_depth is not None and child_depth >= self.outer_tree_max_depth:
                    continue
                if len(child_conjunctions) < self.min_conjunctions_split:
                    continue
                assert isinstance(child_conjunctions[0], Conjunction), 'there we go'
                c_col, c_idx, c_fbt, c_result, conjs_split = self._best_feature_split(child_conjunctions, feature_cols, label_col)
                if c_col is None:
                    continue
                c_left, c_right = conjs_split
                heapq.heappush(heap, (-c_result['impurity_decrease'], child_idx, c_col, c_idx, c_fbt, c_result, (c_left, c_right))) # ADD CHILD TO THE HEAP
            
            self.shape_n_leaves = int(np.sum(self.shape_is_leaf))

        print(f'ShapeFBT: TDIDT complete -- {self.shape_n_leaves} leaves, '
              f'{len(self.shape_nodes)} total nodes.')
        return self

    def prune(self,train):
        """
        :param train: pandas dataframe used as a pruning dataset
        :return: creates a pruned decision forest (include only the relevant trees)
        """
        if self.pruning_method == None:
            self.trees_conjunctions = self.trees_conjunctions_total
        self.pruner = Pruner()
        if self.pruning_method == 'auc':
            self.trees_conjunctions = self.pruner.max_auc_pruning(self.trees_conjunctions_total, train[self.feature_cols],
                                                                      train[self.label_col], min_forest_size=self.min_forest_size)

    def _route_instance(self, x):
        """Route a single raw instance through the outer and inner trees."""
        node_idx = 0
        while self.shape_children[node_idx]:  # while not a leaf
            feat_col, feat_idx, fbt = self.shape_nodes[node_idx]
            
            # walk the inner FBT tree on this one feature
            inner_node = fbt.tree
            while inner_node.selected_feature is not None:
                if x[feat_idx] >= inner_node.selected_value: # based on tree.predict_proba_and_depth
                    inner_node = inner_node.left
                else:
                    inner_node = inner_node.right
            
            # inner_node is now a leaf with bucket_assignment set by map_to_buckets
            branch = inner_node.bucket_assignment
            branch = min(branch, len(self.shape_children[node_idx]) - 1)
            node_idx = self.shape_children[node_idx][branch]

        return _leaf_proba(self.shape_conjunctions[node_idx])
    def predict_Xy(self, X, y=None):
        """
        Route raw inputs through the outer shape tree and inner FBT trees.

        NOTE: Returns a 2-tuple — always unpack as: predictions, probas = model.predict_Xy(X)

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray, shape (n_samples, n_features)
        y : array-like, optional — if provided, also prints accuracy

        Returns
        -------
        tuple of:
            predictions : np.ndarray of int, shape (n_samples,)  — argmax class indices
            probas      : np.ndarray of float, shape (n_samples, n_classes)  — class probabilities
        """
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values

        probas = np.array([self._route_instance(x) for x in X])
        predictions = np.argmax(probas, axis=1)

        if y is not None:
            acc = np.mean(predictions == np.array(y))
            print(f'Accuracy: {acc:.4f}')

        return predictions, probas



    # PREDICTION - OLDDDD
    def _predict_instance_proba(self, inst, node_idx):
        if self.shape_is_leaf[node_idx]:
            conjs = self.shape_conjunctions[node_idx]
            if not conjs:
                import warnings
                warnings.warn(
                    f'Leaf node {node_idx} has no conjunctions; falling back to root conjunctions. '
                    'This may indicate a split routed all conjunctions to one side.',
                    RuntimeWarning, stacklevel=2
                )
                conjs = self.shape_conjunctions[0]
            return _leaf_proba(conjs)
        feat_col, feat_idx, fbt = self.shape_nodes[node_idx]
        branch = int(_route_through_buckets(fbt, inst.reshape(1, -1), feat_idx)[0])
        branch = min(branch, len(self.shape_children[node_idx]) - 1)
        return self._predict_instance_proba(inst, self.shape_children[node_idx][branch])

    def predict_proba(self, X):
        """Return class probability estimates, shape (n_samples, n_classes)."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        return np.array([self._predict_instance_proba(inst, 0) for inst in X])

    def predict(self, X):
        """Return predicted class indices."""
        return np.argmax(self.predict_proba(X), axis=1)

    def get_decision_paths(self, X): # """Return a human-readable decision path per instance."""
        if isinstance(X, pd.DataFrame):
            X = X[self.feature_cols].values
        paths = []
        for inst in X:
            path, node_idx = [], 0
            while not self.shape_is_leaf[node_idx]:
                feat_col, feat_idx, fbt = self.shape_nodes[node_idx]
                branch = int(_route_through_buckets(fbt, inst.reshape(1, -1), feat_idx)[0])
                branch = min(branch, len(self.shape_children[node_idx]) - 1)
                path.append(
                    f'{feat_col} -> branch {branch} '
                    f'(depth {self.shape_depths[node_idx]})'
                )
                node_idx = self.shape_children[node_idx][branch]
            conjs = self.shape_conjunctions[node_idx]
            if not conjs:
                import warnings
                warnings.warn(
                    f'Leaf node {node_idx} has no conjunctions; falling back to root conjunctions.',
                    RuntimeWarning, stacklevel=2
                )
                conjs = self.shape_conjunctions[0]
            path.append(
                f'leaf {node_idx} | {len(self.shape_conjunctions[node_idx])} conjunctions '
                f'| pred={int(np.argmax(_leaf_proba(conjs)))}'
            )
            paths.append(path)
        return paths

