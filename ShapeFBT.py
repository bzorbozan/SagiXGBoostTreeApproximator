""""
This module contains the ShapeFBT which acts as the outer tree in our Sagi-ShapeCART Algorithm
"""
from conjunctionset import *
from tree import *
from pruning import *
from FBT import *



class ShapeFBT():
    def __init__():

        # Attributes needed to calculate conjunction sets
        self.pruning_method = pruning_method
        self.min_forest_size = min_forest_size
        self.max_number_of_conjunctions = max_number_of_conjunctions
        self.pruning_method = pruning_method
        self.max_depth = max_depth

        # Related to inner loop
        self.min_impurity_decrease = min_impurity_decrease
        self.k = k

        # Related to outer loop
    
    def _fit_one_feature(self, feat_col, label_col):
        """
        Fit FBT on a single feature at the current node, then map_to_buckets.

        Uses the splitting points of one feature only, forcing Tree.split() to only iterate thorugh that feature's values

        Returns (fbt, result) or (None, None) if no valid split.
        """
        fbt = FBT(
            max_depth=self.max_depth,
            min_forest_size=self.min_forest_size,
            max_number_of_conjunctions=self.max_number_of_conjunctions,
            pruning_method=None,
        )
        try:
            splitting_points_one_feature =  {feat_col: self.cs.splitting_points[feat_col]} if feat_col in self.cs.splitting_points else {}
            fbt.fit( 
                self.cs.conjunctions, 
                splitting_points_one_feature, # this will force Tree.split() to only go through the key-value pairs of one feature
                feature_cols=[feat_col],
                label_col=label_col
            )
            result = fbt.map_to_buckets(k=self.k)
        except Exception:
            return None, None

        if result['impurity_decrease'] <= self.min_impurity_decrease:
            return None, None

        return fbt, result

    def _best_feature_split(self, train_node, feature_cols, label_col):
        """
        Call FBT.fit() + map_to_buckets() for every feature on the node's
        data. Return the feature with the highest impurity decrease.

        Returns (feat_col, feat_idx, fbt, result)
        or      (None, None, None, None) if no valid split.
        """
        best_imp  = self.min_impurity_decrease
        best      = (None, None, None, None)

        for feat_idx, feat_col in enumerate(feature_cols):

            fbt, result = self._fit_one_feature(feat_col, label_col)

            if fbt is not None and result['impurity_decrease'] > best_imp:
                best_imp = result['impurity_decrease']
                best     = (feat_col, feat_idx, fbt, result)

        return best

    # def _select_best_feature():
    def fit(self,train,feature_cols,label_col, xgb_model, pruned_forest=None, trees_conjunctions_total=None):
        # Everything until this line (until next comment) shoud be taken out and calculated in the outer loop
        self.feature_cols = feature_cols
        self.label_col = label_col
        self.int_cols = [k for k,v in train[feature_cols].dtypes.items() if 'int' in str(v)]
        self.xgb_model = xgb_model
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
        # Everything above this line (below previous comment) should be taken out and calculated in the outer loop
        # self.tree = Tree(self.cs.conjunctions, self.cs.splitting_points,self.max_depth)
        # self.tree.split()

        # Now we pass this into the FBT without needing to re-calculate it during every loop
        # we will have splitting_points only be called for a single feature (type still has to be a dict)
        # honestly you can just call the rest directly, it is just 2 lines. but still keep there for modularity and to not mess with it as much 
        # then you call map to buckets which give you the L/R results -> 0/1 

        while ___:
            Run CART with FBT

        # ── Step 2: best-first TDIDT loop ────────────────────────────────
        print('ShapeFBT: starting TDIDT loop ...')
        X = train[feature_cols].values
        y = train[label_col].values

        # initialise root (node 0)
        self.shape_nodes        = [None]
        self.shape_conjunctions = [self.cs.conjunctions]
        self.shape_point_idxs   = [np.arange(len(y))]
        self.shape_children     = [[]]
        self.shape_parents      = [None]
        self.shape_depths       = [0]
        self.shape_is_leaf      = [True]
        self.shape_n_leaves     = 1

        feat_col, feat_idx, fbt, result = self._best_feature_split(
            train, feature_cols, label_col
        )
        if feat_col is None:
            print('ShapeFBT: no valid split at root -- single-leaf tree.')
            return self

        heap = []
        heapq.heappush(
            heap, (-result['impurity_decrease'], 0, feat_col, feat_idx, fbt, result)
        )

        while heap:
            neg_imp, node_idx, feat_col, feat_idx, fbt, result = heapq.heappop(heap)

            if self.max_depth is not None and self.shape_depths[node_idx] >= self.max_depth:
                continue

            node_point_idxs   = self.shape_point_idxs[node_idx]
            node_conjunctions = self.shape_conjunctions[node_idx]
            X_node            = X[node_point_idxs]

            if len(node_conjunctions) < self.min_conjunctions_split:
                continue

            # commit as internal node
            self.shape_nodes[node_idx]    = (feat_col, feat_idx, fbt)
            self.shape_is_leaf[node_idx]  = False
            self.shape_children[node_idx] = []

            branching = _route_through_buckets(fbt, X_node, feat_idx)
            k_actual  = int(branching.max()) + 1

            partitions = _partition_conjunctions(
                node_conjunctions, X_node, branching, k_actual
            )

            self._log(
                f'  Node {node_idx} (depth {self.shape_depths[node_idx]}): '
                f'split on "{feat_col}", {k_actual} branches, '
                f'impurity_decrease={-neg_imp:.4f}'
            )

            for branch_idx in range(k_actual):
                child_idx          = len(self.shape_nodes)
                child_depth        = self.shape_depths[node_idx] + 1
                child_point_idxs   = node_point_idxs[branching == branch_idx]
                child_conjunctions = partitions[branch_idx]

                self.shape_nodes.append(None)
                self.shape_conjunctions.append(child_conjunctions)
                self.shape_point_idxs.append(child_point_idxs)
                self.shape_children.append([])
                self.shape_parents.append(node_idx)
                self.shape_depths.append(child_depth)
                self.shape_is_leaf.append(True)
                self.shape_children[node_idx].append(child_idx)

                if len(child_point_idxs) < self.min_samples_split:
                    continue
                if len(child_conjunctions) < self.min_conjunctions_split:
                    continue
                if self.max_depth is not None and child_depth >= self.max_depth:
                    continue
                if len(np.unique(y[child_point_idxs])) == 1:
                    continue

                train_child = train.iloc[child_point_idxs]
                c_col, c_idx, c_fbt, c_result = self._best_feature_split(
                    train_child, feature_cols, label_col
                )
                if c_col is None:
                    continue

                heapq.heappush(
                    heap,
                    (-c_result['impurity_decrease'], child_idx,
                     c_col, c_idx, c_fbt, c_result)
                )

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
                


def _leaf_proba(conjunctions):
    """Mean softmax over conjunctions -- mirrors Tree.predict_instance_proba."""
    return np.array([softmax(c.label_probas) for c in conjunctions]).mean(axis=0)[0]

# BZ NOTE: I am skeptical of this - do we want the conjunctions to eb routed in the outer tree like this?
def _partition_conjunctions(conjunctions, X_node, branching, k):
    """
    Assign each conjunction to one of k child buckets by majority vote
    of the data points it contains at this node.
    Falls back to cosine similarity for conjunctions with no local points.
    """
    partitions = [[] for _ in range(k)]
    for conj in conjunctions:
        inside = np.array([conj.containsInstance(inst) for inst in X_node])
        if inside.any():
            winner = int(np.bincount(branching[inside], minlength=k).argmax())
        else:
            conj_dist = np.exp(conj.label_probas)
            conj_dist /= conj_dist.sum() + 1e-12
            best_b, best_sim = 0, -np.inf
            for b in range(k):
                if not partitions[b]:
                    continue
                b_dist = np.array([
                    np.exp(c.label_probas) / (np.exp(c.label_probas).sum() + 1e-12)
                    for c in partitions[b]
                ]).mean(axis=0)
                sim = np.dot(conj_dist, b_dist) / (
                    np.linalg.norm(conj_dist) * np.linalg.norm(b_dist) + 1e-12)
                if sim > best_sim:
                    best_sim, best_b = sim, b
            winner = best_b
        partitions[winner].append(conj)
    return partitions

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
