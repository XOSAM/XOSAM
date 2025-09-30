import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class ScratchSimpleNeuralNetworkClassifier:
    def __init__(self, n_features, n_hidden, n_output, lr=0.01, n_epochs=50, batch_size=32, random_state=42):
        np.random.seed(random_state)
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        # Xavier initialization
        self.W1 = np.random.randn(n_features, n_hidden) / np.sqrt(n_features)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) / np.sqrt(n_hidden)
        self.b2 = np.zeros((1, n_output))

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return x * (1 - x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def _cross_entropy(self, y_true, y_pred):
        m = y_true.shape[0]
        return -np.sum(y_true * np.log(y_pred + 1e-15)) / m

    def _one_hot(self, y):
        one_hot = np.zeros((y.size, self.n_output))
        one_hot[np.arange(y.size), y] = 1
        return one_hot

    def fit(self, X, y):
        y_one_hot = self._one_hot(y)

        for epoch in range(self.n_epochs):
            for i in range(0, X.shape[0], self.batch_size):
                X_batch = X[i:i+self.batch_size]
                y_batch = y_one_hot[i:i+self.batch_size]

                # Forward pass
                z1 = np.dot(X_batch, self.W1) + self.b1
                a1 = self._sigmoid(z1)
                z2 = np.dot(a1, self.W2) + self.b2
                y_pred = self._softmax(z2)

                # Backward pass
                error2 = y_pred - y_batch
                dW2 = np.dot(a1.T, error2) / X_batch.shape[0]
                db2 = np.sum(error2, axis=0, keepdims=True) / X_batch.shape[0]

                error1 = np.dot(error2, self.W2.T) * self._sigmoid_deriv(a1)
                dW1 = np.dot(X_batch.T, error1) / X_batch.shape[0]
                db1 = np.sum(error1, axis=0, keepdims=True) / X_batch.shape[0]

                # Update weights
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1

            # Loss per epoch
            loss = self._cross_entropy(y_one_hot, self.predict_proba(X))
            print(f"Epoch {epoch+1}/{self.n_epochs} - Loss: {loss:.4f}")

    def predict_proba(self, X):
        z1 = np.dot(X, self.W1) + self.b1
        a1 = self._sigmoid(z1)
        z2 = np.dot(a1, self.W2) + self.b2
        return self._softmax(z2)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


# =========================
# Run on Iris dataset demo
# =========================
if __name__ == "__main__":
    # Load data
    X, y = load_iris(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = ScratchSimpleNeuralNetworkClassifier(
        n_features=4, n_hidden=10, n_output=3, lr=0.1, n_epochs=50, batch_size=16
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print("\nTest Accuracy:", accuracy)
      
