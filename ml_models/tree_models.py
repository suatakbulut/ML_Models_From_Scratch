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
        

class RandomForestClassifier:
    def __init__(self, n_estimators=21, max_depth=10, min_samples_split=2, n_features=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.trees = []
        
    def fit(self, X, y):
        for _ in range(self.n_estimators):
            tree = DecisionTreeClassifier(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split, 
                n_features=self.n_features)
            
            X_sample, y_sample = self._bootstrap(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
        
    def _bootstrap(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    
    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common()[0][0]
    
    def predict(self, X):
        tree_predictions = [tree.predict(X) for tree in self.trees]
        all_preds = np.swapaxes(tree_predictions, 0, 1)
        return [self._most_common_label(pred) for pred in all_preds]


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from sklearn import datasets
    import matplotlib.pyplot as plt
    from metrics import mean_square_error, accuracy 

    bc = datasets.load_breast_cancer()
    X, y = bc.data, bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    ###########################
    # Decision Tree Classifier 
    ###########################
    dtc = DecisionTreeClassifier(max_depth=10, min_samples_split=2)
    dtc.fit(X_train, y_train) 
    y_pred = dtc.predict(X_test)
    print("\nTesting Decision Tree Classifier")
    print("================================")
    print(" Custom Accuracy:", accuracy(y_test, y_pred))


    from sklearn.tree import DecisionTreeClassifier as DTC
    sk_dtc = DTC()
    sk_dtc.fit(X_train, y_train) 
    y_pred = sk_dtc.predict(X_test)
    print("Sklearn Accuracy:", accuracy(y_test, y_pred))


    ###########################
    # Random Forest Classifier 
    ###########################
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train) 
    y_pred = rf.predict(X_test)
    print("\nTesting Random Forest Classifier")
    print("================================")
    print(" Custom Accuracy:", accuracy(y_test, y_pred))

    from sklearn.ensemble import RandomForestClassifier as RF
    sk_rf = RF(n_jobs=-1)
    sk_rf.fit(X_train, y_train) 
    y_pred = sk_rf.predict(X_test)
    print("Sklearn Accuracy:", accuracy(y_test, y_pred))


