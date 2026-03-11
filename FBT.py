"""
This module contains a forest based tree class (FBT).

The class takes an XGBoost as an input and generates a decision aims at preserving the predictive performance of
the XGboost model
"""
from conjunctionset import *
from tree import *
from pruning import *

# New imports
import warnings
from sklearn.cluster import KMeans

class FBT():
    """
    This class creates a decision tree from an XGboost
    """
    def __init__(self,max_depth,min_forest_size,max_number_of_conjunctions,pruning_method=None):
        """

        :param max_depth: Maximum allowed depths of the generated tree
        :param min_forest_size: Minimum size of the pruned forest (relevant for the pruning stage)
        :param max_number_of_conjunctions:
        :param pruning_method: Pruning method. If None then there's no pruning. 'auc' is for greedy auc-bsed pruning
        :param xgb_model: Trained XGboost model
        """
        self.min_forest_size = min_forest_size
        self.max_number_of_conjunctions = max_number_of_conjunctions
        self.pruning_method = pruning_method
        self.max_depth = max_depth

        # Coordinate Descent Attributes (default vals not part of initialization for now)
        self.criterion = 'entropy'
        self.criterion_flag = 1 if self.criterion == 'entropy' else 0
        self.smart_init = True
        self.max_iter = 10
        self.random_state = 42

    def fit(self,conj_set, feature_cols,label_col):
        """
        Generates the decision tree by applying the following stages:
        1. Generating a conjunction set that represents each tree of the decision forest
        2. Prune the decision forest according to the given pruning approach
        3. Generate the conjunction set (stage 1 in the algorithm presented)
        4. Create a decision tree out of the generated conjunction set

        :param train: pandas dataframe that was used for training the XGBoost
        :param feature_cols: feature column names
        :param label_col: label column name
        :param xgb_model: XGBoost
        :param pruned_forest: A list of trees, represnt a post-pruning forest. Relevant mostly for the experiment presented in the paper
        :param tree_conjunctions: This para
        """
        self.feature_cols = feature_cols
        self.label_col = label_col
   
        self.cs = conj_set
        print('Start ordering conjunction set in a tree structure')
        self.tree = Tree(self.cs.conjunctions, self.cs.splitting_points,self.max_depth)
        self.tree.split()
        print('Construction of tree has been completed')

    # def prune(self,train):
    #     """

    #     :param train: pandas dataframe used as a pruning dataset
    #     :return: creates a pruned decision forest (include only the relevant trees)
    #     """
    #     if self.pruning_method == None:
    #         self.trees_conjunctions = self.trees_conjunctions_total
    #     self.pruner = Pruner()
    #     if self.pruning_method == 'auc':
    #         self.trees_conjunctions = self.pruner.max_auc_pruning(self.trees_conjunctions_total, train[self.feature_cols],
    #                                                                   train[self.label_col], min_forest_size=self.min_forest_size)

    def predict_proba(self,X):
        """
        Returns class probabilities

        :param X: Pandas dataframe or a numpy matrix
        :return: class probabilities for the corresponding data
        """
        return self.tree.predict_proba(X)

    def predict(self, X):
        """
        Get predictions vector

        :param X: Pandas dataframe or a numpy matrix
        :return: Predicted classes
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def get_decision_paths(self, X):
        """

        :param X: Pandas data frame of [number_of_instances, number_of_features] dimension
        :return: A list of decision paths where each decision path represented as a string of nodes. one node for the leaf and the other for the decision nodes
        """
        paths = self.tree.get_decision_paths(X)
        processed_paths = []
        for path in paths:
            temp_path = []
            for node in path:
                if node.startswith('label'):
                    temp_path.append(node)
                else:
                    if '<' in node:
                        splitted = node.split('<')
                        temp_path.append(self.feature_cols[int(splitted[0])]+' < '+splitted[1])
                    else:
                        splitted = node.split('>=')
                        temp_path.append(self.feature_cols[int(splitted[0])] + ' >= ' + splitted[1])
            processed_paths.append(temp_path)
        return processed_paths

    #######################################################################
    # SHAPECART RELATED ADJUSTMENTS

    def _get_all_leaves(self, node):
        """Helper: recursively collect all leaf nodes"""
        if node.selected_feature is None:  # Is leaf
            return [node]
        
        leaves = []
        if node.left:
            leaves.extend(self._get_all_leaves(node.left))
        if node.right:
            leaves.extend(self._get_all_leaves(node.right))
        return leaves
    
    def _get_leaf_subtree_sides(self):
        """
        Get which side of the root (left=0 or right=1) each leaf falls on.
        Assumes that leaf.leaf_idx has already been assigned by map_to_buckets
        Returns: np.array of shape (n_leaves,) with values 0 or 1
        """
        if self.tree.selected_feature is None: # tree is just single leaf
            return np.array([0])
        
        leaf_sides_dict = {}
        
        def traverse_and_assign(node, side):
            if node.selected_feature is None:  # is leaf
                # use EXISTING leaf_idx (don't create new one)
                if hasattr(node, 'leaf_idx'):
                    leaf_sides_dict[node.leaf_idx] = side
                return
            
            if node.left:
                traverse_and_assign(node.left, side)
            if node.right:
                traverse_and_assign(node.right, side)
        
        # Traverse subtrees
        if self.tree.left:
            traverse_and_assign(self.tree.left, 0)
        if self.tree.right:
            traverse_and_assign(self.tree.right, 1)
        
        # Convert to array
        n_leaves = len(leaf_sides_dict)
        leaf_sides = np.array([leaf_sides_dict[i] for i in range(n_leaves)], dtype=np.int32)
        
        return leaf_sides

    def map_to_buckets(self, k=2):
        """"
        Implemented for 2 buckets only for now. Will be updated in the future for k-buckets
        """
        # Get all leaves from the tree
        leaves = self._get_all_leaves(self.tree)

        # Assign idx here for consistency & re-use everywhere else -> so that the idx is consistent regardless fo traversal order
        for leaf_idx, leaf in enumerate(leaves):
            leaf.leaf_idx = leaf_idx 
        
        # Get the leaf sides -> in example shapecart code (BranchingTree.py) Leaf Sides only calculated if K==2
        if k == 2: 
            # get init solution from tree
            leaf_sides = self._get_leaf_subtree_sides()
        else:
            leaf_sides = None

        # Prepare data
        leaf_distributions = []
        leaf_samples = []
        leaf_nodes = []
        
        for leaf_idx, leaf in enumerate(leaves):
            probas = np.array([softmax(c.label_probas) for c in leaf.conjunctions]).mean(axis=0).flatten()
            # print(f"Leaf {leaf_idx}: probas shape = {probas.shape}")
            leaf_distributions.append(probas)
            leaf_samples.append(len(leaf.conjunctions))
            leaf_nodes.append(leaf_idx)
        
        leaf_distributions = np.array(leaf_distributions)
        leaf_samples = np.array(leaf_samples)
        leaf_nodes = np.array(leaf_nodes)

        # Run coordinate descent (assuming you have bucketer object)
        result = self.coordinate_descent(
            k_=k,
            leaf_distributions=leaf_distributions,
            leaf_samples=leaf_samples,
            leaf_nodes=leaf_nodes,
            leaf_sides=leaf_sides
        )
        
        # Store results (Note that assignments are integers from 0 to K-1, where K=2 for now) -> range(0,K)
        bucket_assignments = result['mapping']
        
        left_conjunctions, right_conjunctions = [],[] # Bucket the actual conjunctions as left or right (or other k-directions, but for now keep as L/R

        # Assign buckets back to leaves 
        # BZ TO DO: - you can do this better since all leaf samples and assignments are 2 arrays and you can more easily assign to K-branches
        # and then instead of a left right array you can just return an array where each item is a branch [left_conjunctions, middle_conjunctions, right_conjunctions]
        for leaf_idx, leaf in enumerate(leaves):
            leaf.bucket_assignment = bucket_assignments[leaf_idx]
            if leaf.bucket_assignment == 0:
                left_conjunctions.append(leaf.conjunctions)
            else: # bucket_assignment == 1 (go right)
                right_conjunctions.append(leaf.conjunctions)
        return result, left_conjunctions, right_conjunctions

    def coordinate_descent(self, k_, leaf_distributions, leaf_samples, leaf_nodes, leaf_sides=None):

        # run the kmeans algorithm but catch convergence warnings and raise an exception
        if self.smart_init:
            with warnings.catch_warnings():
                warnings.filterwarnings('error')
                try:
                    kmeans = KMeans(n_clusters=k_, copy_x=False)
                    kmeans.fit(leaf_distributions, sample_weight=leaf_samples)
                    assignments = kmeans.labels_
                except Warning:
                    # this means that the # of distinct points is less than k_, we can fallback to random init
                    assignments = np.random.randint(0, k_, size=len(leaf_nodes))
        else:
            assignments = np.random.randint(0, k_, size=len(leaf_nodes))
        leaf_weighted_dists = leaf_distributions * leaf_samples[:, np.newaxis]

        partition_weighted_distributions = np.zeros((k_, leaf_distributions.shape[1]), dtype=np.float64)
        for i in range(k_):
            mask = assignments == i
            weighted_distributions = leaf_weighted_dists[mask]
            partition_weighted_distribution = np.sum(weighted_distributions, axis = 0)
            partition_weighted_distributions[i] = partition_weighted_distribution

        leaf_side_partition_weighted_distributions = np.zeros((k_, leaf_distributions.shape[1]), dtype=np.float64)
        if leaf_sides is not None:
            for i in range(k_):
                mask = leaf_sides == i
                weighted_distributions = leaf_weighted_dists[mask]
                partition_weighted_distribution = np.sum(weighted_distributions, axis = 0)
                leaf_side_partition_weighted_distributions[i] += partition_weighted_distribution

        assignments, partition_weighted_distributions, total_impurity = run_descent(
            len(leaf_nodes),
            assignments,
            leaf_weighted_dists,
            partition_weighted_distributions,
            k_,
            self.criterion_flag, 
            max_iter=self.max_iter, 
            seed = self.random_state, 
            leaf_sides=leaf_sides,
            leaf_side_partition_weighted_distributions=leaf_side_partition_weighted_distributions
        )
        mapping = np.zeros(leaf_nodes.max() + 1, dtype=np.int32)
        for i in range(len(leaf_nodes)):
            mapping[leaf_nodes[i]] = assignments[i]

        # calculate the initial impurity here
        initial_impurity = calculate_total_impurity( np.sum(leaf_weighted_dists, axis=0, keepdims=True), self.criterion_flag) 
        
        impurities = {}
        impurities = {k_val: self.calculate_impurity(partition_weighted_distributions[k_val], weighted = True) for k_val in range(partition_weighted_distributions.shape[0])}
        partition_weights = [np.sum(partition_weighted_distributions[part_]) for part_ in range(k_)]
        impurity_decrease = initial_impurity - total_impurity
        return {
            'weighted_distributions': partition_weighted_distributions,
            'impurities': impurities,
            'impurity': total_impurity,
            'partition_weights': partition_weights,
            'impurity_decrease': impurity_decrease,
            'mapping': mapping,
        }

    def calculate_impurity(self, distribution, weighted = False):
        if weighted:
            if np.sum(distribution) == 0:
                return 0
            distribution = distribution / np.sum(distribution)
        
        if self.criterion == 'gini': # this used to be an attribute in the BranchingTree class in SGTLearn Repo
            return 1 - np.sum(distribution**2)
        elif self.criterion == 'entropy':
            impurity = 0
            for p in distribution:
                if p > 0:
                    impurity -= p * np.log2(p)
            return impurity
        else:
            raise ValueError('Criterion must be either gini or entropy')

    #######################################################################

    #######################################################################
    #The following functions are only relevant for the experiment
    # They should be excluded from the documentation of the package
    ########################################################################

    def predict_proba_and_depth(self,X):
        """
        Get class probabilities and depths for each instance

        :param X: Pandas dataframe or a numpy matrix
        :return: class probabilities and the depth of each prediction
        """
        return self.tree.predict_proba_and_depth(X)

    def predict_proba_pruned_forest(self,X):
        """
        Predict_proba using the pruned forest

        :param X: Pandas dataframe or a numpy matrix
        :return: Class probabilities according to the pruned forest
        """
        return self.pruner.predict_probas(self.trees_conjunctions,X)

    def predict_proba_and_depth_forest(self,X):
        """
                Predict_proba and depth using the original forest

                :param X: Pandas dataframe or a numpy matrix
                :return: Class probabilities according to the forest and corresponding depths
        """
        probas = []
        depths = []
        for inst in X.values:
            proba=[]
            depth = 0
            for t in self.trees_conjunctions_total:
                for conj in t:
                    if conj.containsInstance(inst):
                        depth+= np.sum(np.abs(conj.features_upper)!=np.inf) + np.sum(np.abs(conj.features_lower)!=np.inf)
                        proba.append(conj.label_probas)
            depths.append(depth)
            probas.append(softmax(np.array(proba).sum(axis=0)))
        return np.array([i[0] for i in probas]), depths

    def predict_proba_and_depth_pruned_forest(self,X):
        """
        Predict_proba and depth using the pruned forest

        :param X: Pandas dataframe or a numpy matrix
        :return: Class probabilities according to the pruned forest and corresponding depths
        """
        probas = []
        depths = []
        for inst in X.values:
            proba=[]
            depth = 0
            for t in self.trees_conjunctions:
                for conj in t:
                    if conj.containsInstance(inst):
                        depth+= np.sum(np.abs(conj.features_upper)!=np.inf) + np.sum(np.abs(conj.features_lower)!=np.inf)
                        proba.append(conj.label_probas)
            depths.append(depth)
            probas.append(softmax(np.array(proba).sum(axis=0)))
        return np.array([i[0] for i in probas]), depths

#######################################################################
# SHAPECART RELATED ADJUSTMENTS
def calculate_total_impurity(weighted_dist: np.ndarray, criterion_flag: int) -> float:
    """
    weighted_dist: 2D array of shape (n_partitions, n_classes)
                    each row i is the count-vector for partition i.
    criterion_flag: 0 for Gini, 1 for entropy
    """
    imp = 0.0
    total = 0.0
    n_parts, n_classes = weighted_dist.shape

    for i in range(n_parts):
        # 1) compute sum of counts for this partition
        s = 0.0
        for j in range(n_classes):
            s += weighted_dist[i, j]
        if s == 0.0:
            continue

        if criterion_flag == 0:
            # Gini: sum * (1 - sum_k (p_k^2))
            dot = 0.0
            for j in range(n_classes):
                p = weighted_dist[i, j] / s
                dot += p * p
            imp += s * (1.0 - dot)
        else:
            # Entropy: - sum * sum_k (p_k * log2 p_k)
            e = 0.0
            for j in range(n_classes):
                p = weighted_dist[i, j] / s
                if p > 0.0:
                    e += p * np.log2(p)
            imp -= s * e

        total += s

    if total == 0.0:
        return 0.0
    return imp / total

def run_descent(n_leaf_nodes: int,
                assignments: np.ndarray, # assignments from kmeans
                leaf_weighted_dists: np.ndarray, # leaf_weights
                partition_weighted_distributions: np.ndarray, # left/right weighted dists from kmeans
                k_: int,
                criterion_flag: int,
                max_iter: int = 10,
                seed: int = 42,
                leaf_sides: np.ndarray = None, # assignments from root split
                leaf_side_partition_weighted_distributions: np.ndarray = None # left/right weighted dists from root 
                ):
    counter = 0
    total_impurity = calculate_total_impurity(
        partition_weighted_distributions, criterion_flag=criterion_flag
    )
    if leaf_sides is not None: # if we have leaf sides, check if they are better. if yes, use them as init
        leaf_sides_total_impurity = calculate_total_impurity(
            leaf_side_partition_weighted_distributions, criterion_flag=criterion_flag
        )
        if leaf_sides_total_impurity < total_impurity:
            total_impurity = leaf_sides_total_impurity
            assignments = leaf_sides.copy()
            partition_weighted_distributions = leaf_side_partition_weighted_distributions.copy()
        else:
            leaf_sides_total_impurity = np.inf

    old_total_impurity = total_impurity
    best_impurity = total_impurity
    if max_iter == 0:
        return assignments, partition_weighted_distributions, total_impurity
    rng = np.random.default_rng(seed)
    for _ in range(max_iter):
        leaves = rng.permutation(n_leaf_nodes)
        switched = False
        for leaf in leaves:
            curr_assignment = assignments[leaf]
            contr = leaf_weighted_dists[leaf]
            
            # Temporarily remove contribution of current leaf
            partition_weighted_distributions[curr_assignment] -= contr
            
            best_assignment = curr_assignment
            
            for k_val in range(k_):
                if k_val == curr_assignment: # skip the current assignment as this is the best impurity
                    continue
                # Temporarily add leaf's contribution to new candidate partition
                partition_weighted_distributions[k_val] += contr
                
                # Compute impurity only if candidate partition is changed
                new_impurity = calculate_total_impurity(partition_weighted_distributions, criterion_flag=criterion_flag)
                
                if new_impurity < best_impurity:
                    best_impurity = new_impurity
                    best_assignment = k_val
                    switched = True
                
                # Restore partition distribution
                partition_weighted_distributions[k_val] -= contr
            
            # Permanently assign the leaf to the best partition
            assignments[leaf] = best_assignment
            partition_weighted_distributions[best_assignment] += contr
            total_impurity = best_impurity  # Update total_impurity directly
        assert total_impurity <= old_total_impurity, f'Total impurity increased: {total_impurity} > {old_total_impurity}'
        if not switched:
            counter += 1
        else:
            counter = 0
        
        if counter > 5:
            break

    return assignments, partition_weighted_distributions, total_impurity

#######################################################################
