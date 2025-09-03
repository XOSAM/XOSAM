import numpy as np

class ScratchKMeans:
    """
    K-means scratch implementation
    """
    def __init__(self, n_clusters=3, n_init=10, max_iter=300, tol=1e-4, verbose=False):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.centroids = None
        self.inertia_ = None  # SSE

    def _initialize_centroids(self, X):
        # Randomly select n_clusters points from X as initial centroids
        indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
        return X[indices]

    def _assign_clusters(self, X, centroids):
        # Assign each point to the nearest centroid
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        # Compute mean of points assigned to each cluster
        new_centroids = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            if np.any(labels == k):
                new_centroids[k] = X[labels == k].mean(axis=0)
            else:  # avoid empty clusters
                new_centroids[k] = X[np.random.choice(X.shape[0])]
        return new_centroids

    def _compute_sse(self, X, labels, centroids):
        # Sum of squared errors
        sse = 0.0
        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            sse += np.sum((cluster_points - centroids[k])**2)
        return sse

    def fit(self, X):
        best_sse = np.inf
        best_centroids = None
        best_labels = None

        for init in range(self.n_init):
            centroids = self._initialize_centroids(X)
            for i in range(self.max_iter):
                labels = self._assign_clusters(X, centroids)
                new_centroids = self._update_centroids(X, labels)
                shift = np.linalg.norm(new_centroids - centroids)
                centroids = new_centroids
                if shift <= self.tol:
                    break
            sse = self._compute_sse(X, labels, centroids)
            if self.verbose:
                print(f"Init {init+1}, SSE={sse}")
            if sse < best_sse:
                best_sse = sse
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.inertia_ = best_sse
        self.labels_ = best_labels
        if self.verbose:
            print("Best SSE:", self.inertia_)

    def predict(self, X):
        # Assign new data points to nearest centroids
        return self._assign_clusters(X, self.centroids)
