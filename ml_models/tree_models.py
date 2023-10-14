from collections import Counter

import numpy as np


class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecisionTreeClassifier:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X=None, y=None):
        self.n_features = X.shape[1] if not self.n_features else min(
            X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        # build self.root recursively

        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria
        if (n_labels == 1 or depth >= self.max_depth or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # find the best split
        feature_idxs = np.random.choice(
            n_feats, self.n_features, replace=False)
        split_feature, split_threshold = self._best_split(X, y, feature_idxs)

        # create children
        left_idxs, right_idxs = self._split(
            X[:, split_feature], split_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)

        return Node(split_feature, split_threshold, left, right)

    def _best_split(self, X, y, feature_idxs):
        split_feature, split_threshold = None, None
        best_gain = -1
        for feature_idx in feature_idxs:
            X_feature = X[:, feature_idx]
            thresholds = np.unique(X_feature)

            for threshold in thresholds:
                left_idxs, right_idxs = self._split(X_feature, threshold)
                gain = self._information_gain(y, left_idxs, right_idxs)

                if gain > best_gain:
                    best_gain = gain
                    split_feature = feature_idx
                    split_threshold = threshold

        return split_feature, split_threshold

    def _split(self, X_feature, threshold):
        left_idxs = np.argwhere(X_feature <= threshold).flatten()
        right_idxs = np.argwhere(X_feature > threshold).flatten()
        return left_idxs, right_idxs

    def _information_gain(self, y, left_idxs, right_idxs):
        # parent entropy
        parent_entropy = self._entropy(y)
        # average children entropy
        left_entropy = self._entropy(y[left_idxs])
        right_entropy = self._entropy(y[right_idxs])
        n_l, n_r, n = len(left_idxs), len(right_idxs), len(y)
        avg_children_entropy = (n_l/n) * left_entropy + (n_r/n) * right_entropy
        # info gain
        information_gain = parent_entropy - avg_children_entropy
        return information_gain

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p*np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common()[0][0]

    def predict(self, X=None):
        return [self._traverse_tree(x, self.root) for x in X]

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)