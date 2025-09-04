import numpy as np
from sklearn.model_selection import train_test_split

try:
    from tensorflow.keras.datasets import mnist
except Exception:
    from keras.datasets import mnist

class SimpleInitializer:
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def W(self, n1, n2):
        return self.sigma * np.random.randn(n1, n2)
    def B(self, n2):
        return np.zeros(n2)

class XavierInitializer:
    def W(self, n1, n2):
        sigma = 1.0 / np.sqrt(n1)
        return sigma * np.random.randn(n1, n2)
    def B(self, n2):
        return np.zeros(n2)

class HeInitializer:
    def W(self, n1, n2):
        sigma = np.sqrt(2.0 / n1)
        return sigma * np.random.randn(n1, n2)
    def B(self, n2):
        return np.zeros(n2)

class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr
    def update(self, layer):
        layer.W -= self.lr * layer.dW
        layer.B -= self.lr * layer.dB
        return layer

class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.hW = None
        self.hB = None
    def update(self, layer):
        if self.hW is None:
            self.hW = np.zeros_like(layer.W)
            self.hB = np.zeros_like(layer.B)
        self.hW += layer.dW ** 2
        self.hB += layer.dB ** 2
        layer.W -= self.lr * layer.dW / (np.sqrt(self.hW) + 1e-7)
        layer.B -= self.lr * layer.dB / (np.sqrt(self.hB) + 1e-7)
        return layer

class Sigmoid:
    def forward(self, A):
        self.out = 1.0 / (1.0 + np.exp(-A))
        return self.out
    def backward(self, dZ):
        return dZ * self.out * (1.0 - self.out)

class Tanh:
    def forward(self, A):
        self.out = np.tanh(A)
        return self.out
    def backward(self, dZ):
        return dZ * (1.0 - self.out ** 2)

class ReLU:
    def forward(self, A):
        self.mask = A > 0
        return A * self.mask
    def backward(self, dZ):
        return dZ * self.mask

class SoftmaxWithLoss:
    def forward(self, A, y):
        A = A - np.max(A, axis=1, keepdims=True)
        expA = np.exp(A)
        self.probs = expA / np.sum(expA, axis=1, keepdims=True)
        self.y = y
        m = y.shape[0]
        log_likelihood = -np.log(self.probs[np.arange(m), y] + 1e-15)
        self.loss = np.mean(log_likelihood)
        return self.loss
    def backward(self):
        m = self.y.shape[0]
        dA = self.probs.copy()
        dA[np.arange(m), self.y] -= 1.0
        dA /= m
        return dA

class FC:
    def __init__(self, n1, n2, initializer, optimizer):
        self.optimizer = optimizer
        self.W = initializer.W(n1, n2)
        self.B = initializer.B(n2)
    def forward(self, X):
        self.X = X
        return np.dot(X, self.W) + self.B
    def backward(self, dA):
        m = self.X.shape[0]
        self.dW = np.dot(self.X.T, dA)
        self.dB = np.sum(dA, axis=0)
        dZ = np.dot(dA, self.W.T)
        self.optimizer.update(self)
        return dZ

class ScratchDeepNeuralNetworkClassifier:
    def __init__(self, n_features, layer_sizes, activations, initializer="xavier", optimizer="sgd", lr=0.01, sigma=0.01, batch_size=64, epochs=10, verbose=True, seed=42):
        np.random.seed(seed)
        self.n_features = n_features
        self.layer_sizes = layer_sizes
        self.activations_cfg = activations
        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        if initializer == "simple":
            self.initializer_factory = lambda: SimpleInitializer(sigma)
        elif initializer == "he":
            self.initializer_factory = HeInitializer
        else:
            self.initializer_factory = XavierInitializer
        if optimizer == "adagrad":
            self.optimizer_factory = lambda: AdaGrad(lr)
        else:
            self.optimizer_factory = lambda: SGD(lr)
        self._build()

    def _act_from_name(self, name):
        if isinstance(name, str):
            key = name.lower()
            if key == "relu": return ReLU()
            if key == "tanh": return Tanh()
            if key == "sigmoid": return Sigmoid()
            raise ValueError(f"Unknown activation: {name}")
        return name

    def _build(self):
        sizes = [self.n_features] + self.layer_sizes
        self.layers = []
        self.activations = []
        init = self.initializer_factory()
        opt = self.optimizer_factory()
        for i in range(len(sizes) - 1):
            self.layers.append(FC(sizes[i], sizes[i+1], init, opt))
            self.activations.append(self._act_from_name(self.activations_cfg[i]))
        self.softmax = SoftmaxWithLoss()

    def fit(self, X, y, X_val=None, y_val=None):
        for epoch in range(1, self.epochs + 1):
            idx = np.random.permutation(X.shape[0])
            X, y = X[idx], y[idx]
            for i in range(0, X.shape[0], self.batch_size):
                xb = X[i:i+self.batch_size]
                yb = y[i:i+self.batch_size]
                out = xb
                for fc, act in zip(self.layers, self.activations):
                    out = act.forward(fc.forward(out))
                loss = self.softmax.forward(out, yb)
                grad = self.softmax.backward()
                for fc, act in zip(self.layers[::-1], self.activations[::-1]):
                    grad = fc.backward(act.backward(grad))
            if self.verbose:
                msg = f"Epoch {epoch}/{self.epochs} - loss: {loss:.4f}"
                if X_val is not None and y_val is not None:
                    acc = self.score(X_val, y_val)
                    msg += f" - val_acc: {acc:.4f}"
                print(msg)

    def predict(self, X):
        out = X
        for fc, act in zip(self.layers, self.activations):
            out = act.forward(fc.forward(out))
        probs = out - np.max(out, axis=1, keepdims=True)
        probs = np.exp(probs)
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        return np.argmax(probs, axis=1)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
    X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)

    model = ScratchDeepNeuralNetworkClassifier(
        n_features=784,
        layer_sizes=[256, 128, 10],
        activations=["relu", "relu", "identity"],  # last will be fed to softmax
        initializer="he",
        optimizer="adagrad",
        lr=0.05,
        batch_size=128,
        epochs=10,
        verbose=True,
        seed=42
    )

    model.fit(X_tr, y_tr, X_val, y_val)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)
    print(f"Validation Accuracy: {val_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
      
