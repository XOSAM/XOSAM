import numpy as np

# -----------------------------
# Scratch RNN Class (Forward + Backprop)
# -----------------------------
class ScratchSimpleRNNClassifier:
    def __init__(self, n_features, n_nodes, n_output, learning_rate=0.01):
        self.n_features = n_features
        self.n_nodes = n_nodes
        self.n_output = n_output
        self.lr = learning_rate
        
        # Initialize weights
        self.Wx = np.random.randn(n_features, n_nodes) * 0.01
        self.Wh = np.random.randn(n_nodes, n_nodes) * 0.01
        self.bh = np.zeros(n_nodes)
        
        self.Wy = np.random.randn(n_nodes, n_output) * 0.01
        self.by = np.zeros(n_output)
    
    def forward(self, X):
        batch_size, n_sequences, _ = X.shape
        self.cache = {'h': [], 'x': []}
        h_prev = np.zeros((batch_size, self.n_nodes))
        
        for t in range(n_sequences):
            xt = X[:, t, :]
            at = np.dot(xt, self.Wx) + np.dot(h_prev, self.Wh) + self.bh
            ht = np.tanh(at)
            self.cache['h'].append(ht)
            self.cache['x'].append(xt)
            h_prev = ht
        
        self.h_last = ht
        logits = np.dot(ht, self.Wy) + self.by
        self.y_pred = self.softmax(logits)
        return self.y_pred
    
    def softmax(self, z):
        e_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e_z / np.sum(e_z, axis=1, keepdims=True)
    
    def compute_loss(self, y_true):
        return -np.mean(np.sum(y_true * np.log(self.y_pred + 1e-8), axis=1))
    
    def backward(self, y_true):
        batch_size = y_true.shape[0]
        dWy = np.dot(self.cache['h'][-1].T, (self.y_pred - y_true)) / batch_size
        dby = np.sum(self.y_pred - y_true, axis=0) / batch_size
        
        dWx = np.zeros_like(self.Wx)
        dWh = np.zeros_like(self.Wh)
        dbh = np.zeros_like(self.bh)
        dh_next = np.dot((self.y_pred - y_true), self.Wy.T)
        
        for t in reversed(range(len(self.cache['h']))):
            ht = self.cache['h'][t]
            xt = self.cache['x'][t]
            dt = dh_next * (1 - ht**2)  # derivative of tanh
            dWx += np.dot(xt.T, dt)
            dWh += np.dot(self.cache['h'][t-1].T if t > 0 else np.zeros_like(ht).T, dt)
            dbh += np.sum(dt, axis=0)
            dh_next = np.dot(dt, self.Wh.T)
        
        # Gradient descent update
        self.Wx -= self.lr * dWx
        self.Wh -= self.lr * dWh
        self.bh -= self.lr * dbh
        self.Wy -= self.lr * dWy
        self.by -= self.lr * dby
    
    def fit(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.forward(X)
            loss = self.compute_loss(y)
            self.backward(y)
            if (epoch + 1) % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
                
    def predict(self, X):
        probs = self.forward(X)
        return np.argmax(probs, axis=1)

# -----------------------------
# 1. Create Small Synthetic Dataset
# -----------------------------
# 3 sequences, 2 features, 2 classes
X_train = np.array([
    [[0.01, 0.02], [0.02, 0.03], [0.03, 0.04]],
    [[0.05, 0.01], [0.06, 0.02], [0.07, 0.03]],
    [[0.02, 0.05], [0.03, 0.06], [0.04, 0.07]]
])
y_train = np.array([
    [1, 0],  # Class 0
    [0, 1],  # Class 1
    [1, 0]   # Class 0
])

# -----------------------------
# 2. Initialize and Train RNN
# -----------------------------
n_features = X_train.shape[2]
n_nodes = 4
n_output = y_train.shape[1]
rnn = ScratchSimpleRNNClassifier(n_features, n_nodes, n_output, learning_rate=0.1)

print("=== Training on Synthetic Dataset ===")
rnn.fit(X_train, y_train, epochs=50)

# -----------------------------
# 3. Predictions
# -----------------------------
preds = rnn.predict(X_train)
print("\nPredicted classes:", preds)
print("True classes:", np.argmax(y_train, axis=1))
      
