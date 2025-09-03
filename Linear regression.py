import numpy as np
import matplotlib.pyplot as plt

def MSE(y_pred, y):
    mse = np.mean((y_pred - y) ** 2)
    return mse

class ScratchLinearRegression:
    def __init__(self, num_iter=1000, lr=0.01, no_bias=False, verbose=False):
        self.iter = num_iter
        self.lr = lr
        self.no_bias = no_bias
        self.verbose = verbose
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)
        self.coef_ = None

    def _linear_hypothesis(self, X):
        if not self.no_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

    def _gradient_descent(self, X, error):
        grad = np.dot(X.T, error) / X.shape[0]
        self.coef_ -= self.lr * grad

    def fit(self, X, y, X_val=None, y_val=None):
        n_features = X.shape[1]
        if not self.no_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
            if X_val is not None:
                X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
        self.coef_ = np.zeros(n_features + (0 if self.no_bias else 1))
        for i in range(self.iter):
            y_pred = np.dot(X, self.coef_)
            error = y_pred - y
            self._gradient_descent(X, error)
            self.loss[i] = np.mean(error ** 2) / 2
            if X_val is not None and y_val is not None:
                val_pred = np.dot(X_val, self.coef_)
                self.val_loss[i] = np.mean((val_pred - y_val) ** 2) / 2
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: loss={self.loss[i]:.6f} val_loss={self.val_loss[i] if X_val is not None else 'NA'}")

    def predict(self, X):
        if not self.no_bias:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        return np.dot(X, self.coef_)

    def plot_learning_curve(self):
        plt.plot(range(self.iter), self.loss, label='Train Loss')
        if np.any(self.val_loss):
            plt.plot(range(self.iter), self.val_loss, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()




#how to use it 
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = ScratchLinearRegression(num_iter=1000, lr=0.01, verbose=True)
model.fit(X_train, y_train, X_val, y_val)

y_pred = model.predict(X_val)
print("MSE on validation:", MSE(y_pred, y_val))

model.plot_learning_curve()
