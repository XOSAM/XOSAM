import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_loss(y_pred, y_true, coef=None, lam=0.0):
    m = len(y_true)
    eps = 1e-15
    y_pred = np.clip(y_pred, eps, 1 - eps)
    loss = (-1/m) * np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    if coef is not None and lam > 0:
        loss += (lam / (2 * m)) * np.sum(coef[1:] ** 2)
    return loss

class ScratchLogisticRegression:
    def __init__(self, num_iter=1000, lr=0.01, bias=True, verbose=False, lam=0.0):
        self.iter = num_iter
        self.lr = lr
        self.bias = bias
        self.verbose = verbose
        self.lam = lam
        self.coef_ = None
        self.loss = np.zeros(self.iter)
        self.val_loss = np.zeros(self.iter)

    def _linear_hypothesis(self, X):
        if self.bias:
            X = np.hstack([np.ones((X.shape[0],1)), X])
        return np.dot(X, self.coef_)

    def _gradient_descent(self, X, y):
        m = X.shape[0]
        y_pred = sigmoid(np.dot(X, self.coef_))
        error = y_pred - y
        grad = np.dot(X.T, error) / m
        if self.lam > 0:
            grad[1:] += (self.lam / m) * self.coef_[1:]
        self.coef_ -= self.lr * grad

    def fit(self, X, y, X_val=None, y_val=None):
        n_features = X.shape[1]
        if self.bias:
            X = np.hstack([np.ones((X.shape[0],1)), X])
            if X_val is not None:
                X_val = np.hstack([np.ones((X_val.shape[0],1)), X_val])
        self.coef_ = np.zeros(n_features + (0 if not self.bias else 1))
        for i in range(self.iter):
            self._gradient_descent(X, y)
            y_train_pred = sigmoid(np.dot(X, self.coef_))
            self.loss[i] = logistic_loss(y_train_pred, y, self.coef_, self.lam)
            if X_val is not None and y_val is not None:
                y_val_pred = sigmoid(np.dot(X_val, self.coef_))
                self.val_loss[i] = logistic_loss(y_val_pred, y_val, self.coef_, self.lam)
            if self.verbose and i % 100 == 0:
                print(f"Iteration {i}: train_loss={self.loss[i]:.6f} val_loss={self.val_loss[i] if X_val is not None else 'NA'}")

    def predict_proba(self, X):
        if self.bias:
            X = np.hstack([np.ones((X.shape[0],1)), X])
        return sigmoid(np.dot(X, self.coef_))

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

    def plot_learning_curve(self):
        plt.plot(range(self.iter), self.loss, label='Train Loss')
        if np.any(self.val_loss):
            plt.plot(range(self.iter), self.val_loss, label='Validation Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()
