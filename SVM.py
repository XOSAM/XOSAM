import numpy as np

class ScratchSVMClassifier:
    def __init__(self, num_iter=1000, lr=0.001, kernel='linear', threshold=1e-5, verbose=False, gamma=1.0, theta0=0.0, degree=2):
        self.iter = num_iter
        self.lr = lr
        self.kernel = kernel
        self.threshold = threshold
        self.verbose = verbose
        self.gamma = gamma
        self.theta0 = theta0
        self.degree = degree
        
        self.lam_sv = None
        self.X_sv = None
        self.y_sv = None
        self.n_support_vectors = 0
        self.index_support_vectors = None

    def _kernel_func(self, x_i, x_j):
        if self.kernel == 'linear':
            return np.dot(x_i, x_j)
        elif self.kernel == 'polynomial':
            return (self.gamma * np.dot(x_i, x_j) + self.theta0) ** self.degree
        else:
            raise ValueError("Unknown kernel type")

    def _update_lagrange(self, X, y, lam):
        n_samples = X.shape[0]
        for i in range(n_samples):
            K_sum = 0
            for j in range(n_samples):
                K_sum += lam[j] * y[i] * y[j] * self._kernel_func(X[i], X[j])
            lam_new = lam[i] + self.lr * (1 - K_sum)
            lam[i] = max(0, lam_new)
        return lam

    def _determine_support_vectors(self, X, y, lam):
        support_idx = np.where(lam > self.threshold)[0]
        self.n_support_vectors = len(support_idx)
        self.index_support_vectors = support_idx
        self.X_sv = X[support_idx]
        self.y_sv = y[support_idx]
        self.lam_sv = lam[support_idx]

    def fit(self, X, y, X_val=None, y_val=None):
        n_samples = X.shape[0]
        lam = np.zeros(n_samples)
        for it in range(self.iter):
            lam = self._update_lagrange(X, y, lam)
            if self.verbose and it % 100 == 0:
                print(f"Iteration {it} completed")
        self._determine_support_vectors(X, y, lam)

    def predict(self, X):
        n_samples = X.shape[0]
        y_pred = np.zeros(n_samples)
        for i in range(n_samples):
            s = 0
            for n in range(self.n_support_vectors):
                s += self.lam_sv[n] * self.y_sv[n] * self._kernel_func(X[i], self.X_sv[n])
            y_pred[i] = np.sign(s)
        return y_pred
  
