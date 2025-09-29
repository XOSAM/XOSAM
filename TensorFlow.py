import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf = tf.compat.v1

# Load Iris dataset
df = pd.read_csv("Iris.csv")
X = df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']].values.astype(np.float32)
y_raw = df['Species'].values

# Encode labels
le = LabelEncoder()
y_int = le.fit_transform(y_raw)
num_classes = len(le.classes_)
y = np.eye(num_classes)[y_int].astype(np.float32)

# Split dataset
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y_int)
X_train, X_val, y_train, y_val = train_test_split(X_train_full, y_train_full, test_size=0.2, random_state=0)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Mini-batch iterator
class GetMiniBatch:
    def __init__(self, X, y, batch_size=16, seed=0):
        self.batch_size = batch_size
        np.random.seed(seed)
        shuffle_index = np.random.permutation(np.arange(X.shape[0]))
        self.X = X[shuffle_index]
        self.y = y[shuffle_index]
        self._stop = int(np.ceil(X.shape[0] / self.batch_size))
    def __iter__(self):
        self._counter = 0
        return self
    def __next__(self):
        if self._counter >= self._stop:
            raise StopIteration()
        p0 = self._counter * self.batch_size
        p1 = p0 + self.batch_size
        self._counter += 1
        return self.X[p0:p1], self.y[p0:p1]

# Hyperparameters
lr = 0.01
batch_size = 16
num_epochs = 200
n_input = X_train.shape[1]
n_hidden1 = 50
n_hidden2 = 50

# Placeholders
X_ph = tf.placeholder(tf.float32, [None, n_input])
Y_ph = tf.placeholder(tf.float32, [None, num_classes])

# Neural network
def net(x):
    W1 = tf.Variable(tf.random_normal([n_input, n_hidden1]))
    b1 = tf.Variable(tf.random_normal([n_hidden1]))
    W2 = tf.Variable(tf.random_normal([n_hidden1, n_hidden2]))
    b2 = tf.Variable(tf.random_normal([n_hidden2]))
    W3 = tf.Variable(tf.random_normal([n_hidden2, num_classes]))
    b3 = tf.Variable(tf.random_normal([num_classes]))
    l1 = tf.nn.relu(tf.matmul(x, W1) + b1)
    l2 = tf.nn.relu(tf.matmul(l1, W2) + b2)
    logits = tf.matmul(l2, W3) + b3
    return logits

logits = net(X_ph)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_ph, logits=logits))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss_op)
preds = tf.argmax(tf.nn.softmax(logits), axis=1)
labels = tf.argmax(Y_ph, axis=1)
accuracy = tf.reduce_mean(tf.cast(tf.equal(preds, labels), tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(num_epochs):
        mb = GetMiniBatch(X_train, y_train, batch_size=batch_size, seed=epoch)
        for bx, by in mb:
            sess.run(optimizer, feed_dict={X_ph: bx, Y_ph: by})
        if epoch % 20 == 0 or epoch == num_epochs-1:
            val_loss, val_acc = sess.run([loss_op, accuracy], feed_dict={X_ph: X_val, Y_ph: y_val})
            print(f"Epoch {epoch}: val_loss={val_loss:.4f}, val_acc={val_acc:.3f}")
    test_acc = sess.run(accuracy, feed_dict={X_ph: X_test, Y_ph: y_test})
    print("Final test accuracy:", test_acc)
    
