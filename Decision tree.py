import numpy as np

class ScratchDecisionTreeClassifierDepth1:
    """
    Depth 1 decision tree classifier scratch implementation
    """
    def __init__(self, verbose=False):
        self.verbose = verbose
        # Learned tree structure
        self.feature_idx = None
        self.threshold = None
        self.left_class = None
        self.right_class = None

    def _gini_impurity(self, y):
        """
        Calculate Gini impurity for a set of labels y
        """
        classes = np.unique(y)
        N = len(y)
        if N == 0:
            return 0
        impurity = 1.0
        for c in classes:
            p_c = np.sum(y == c) / N
            impurity -= p_c ** 2
        return impurity

    def _information_gain(self, parent_y, left_y, right_y):
        """
        Calculate information gain from splitting parent node into left and right
        """
        N = len(parent_y)
        N_left = len(left_y)
        N_right = len(right_y)
        if N_left == 0 or N_right == 0:
            return 0
        I_parent = self._gini_impurity(parent_y)
        I_left = self._gini_impurity(left_y)
        I_right = self._gini_impurity(right_y)
        IG = I_parent - (N_left / N) * I_left - (N_right / N) * I_right
        return IG

    def fit(self, X, y):
        """
        Learn a depth-1 decision tree
        """
        n_samples, n_features = X.shape
        best_ig = -1

        # Search for best split
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for thresh in thresholds:
                left_idx = X[:, feature] <= thresh
                right_idx = X[:, feature] > thresh
                left_y = y[left_idx]
                right_y = y[right_idx]
                ig = self._information_gain(y, left_y, right_y)
                if ig > best_ig:
                    best_ig = ig
                    self.feature_idx = feature
                    self.threshold = thresh
                    self.left_class = np.bincount(left_y).argmax()
                    self.right_class = np.bincount(right_y).argmax()

        if self.verbose:
            print(f"Depth-1 tree learned: feature {self.feature_idx}, threshold {self.threshold}")
            print(f"Left class: {self.left_class}, Right class: {self.right_class}")

    def predict(self, X):
        """
        Estimate the label using the learned depth-1 decision tree
        """
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples, dtype=int)
        for i in range(n_samples):
            if X[i, self.feature_idx] <= self.threshold:
                y_pred[i] = self.left_class
            else:
                y_pred[i] = self.right_class
        return y_pred
          
