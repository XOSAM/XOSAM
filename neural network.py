import numpy as np

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
        one_hot[np.arange(y.size), y] =_
      
