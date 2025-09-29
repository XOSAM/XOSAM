import tensorflow as tf
from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Enable GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# ----- Prepare datasets -----
# Iris Binary
data = load_iris()
X_bin, y_bin = data.data[data.target != 0], data.target[data.target != 0] - 1
X_bin = StandardScaler().fit_transform(X_bin)

# Iris Multi-class
X_multi, y_multi = data.data, data.target
X_multi = StandardScaler().fit_transform(X_multi)

# House Prices
housing = fetch_california_housing()
X_house, y_house = StandardScaler().fit_transform(housing.data), housing.target

# MNIST
(X_mnist, y_mnist), (X_mnist_test, y_mnist_test) = tf.keras.datasets.mnist.load_data()
X_mnist = X_mnist.astype('float32')/255.0
X_mnist_test = X_mnist_test.astype('float32')/255.0
X_mnist = X_mnist.reshape(-1, 28*28)
X_mnist_test = X_mnist_test.reshape(-1, 28*28)

# Split datasets
X_bin_train, X_bin_test, y_bin_train, y_bin_test = train_test_split(X_bin, y_bin, test_size=0.2, random_state=42)
X_multi_train, X_multi_test, y_multi_train, y_multi_test = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)
X_house_train, X_house_test, y_house_train, y_house_test = train_test_split(X_house, y_house, test_size=0.2, random_state=42)

# ----- Multi-task Model -----
# Input layers
input_bin = tf.keras.Input(shape=(X_bin.shape[1],), name='iris_binary_input')
input_multi = tf.keras.Input(shape=(X_multi.shape[1],), name='iris_multi_input')
input_house = tf.keras.Input(shape=(X_house.shape[1],), name='house_input')
input_mnist = tf.keras.Input(shape=(784,), name='mnist_input')

# Shared hidden layers (optional: separate branches if needed)
def dense_block(x, units=[64, 32]):
    for u in units:
        x = tf.keras.layers.Dense(u, activation='relu')(x)
    return x

# Task-specific outputs
# Iris binary
x_bin = dense_block(input_bin, [32])
out_bin = tf.keras.layers.Dense(1, activation='sigmoid', name='iris_binary_output')(x_bin)

# Iris multi-class
x_multi = dense_block(input_multi, [32])
out_multi = tf.keras.layers.Dense(3, activation='softmax', name='iris_multi_output')(x_multi)

# House Prices regression
x_house = dense_block(input_house, [64, 32])
out_house = tf.keras.layers.Dense(1, name='house_output')(x_house)

# MNIST classification
x_mnist = dense_block(input_mnist, [256, 128])
out_mnist = tf.keras.layers.Dense(10, activation='softmax', name='mnist_output')(x_mnist)

# Model
model = tf.keras.Model(
    inputs=[input_bin, input_multi, input_house, input_mnist],
    outputs=[out_bin, out_multi, out_house, out_mnist]
)

# Compile with multiple losses
model.compile(
    optimizer='adam',
    loss={
        'iris_binary_output': 'binary_crossentropy',
        'iris_multi_output': 'sparse_categorical_crossentropy',
        'house_output': 'mean_squared_error',
        'mnist_output': 'sparse_categorical_crossentropy'
    },
    metrics={
        'iris_binary_output': 'accuracy',
        'iris_multi_output': 'accuracy',
        'house_output': 'mae',
        'mnist_output': 'accuracy'
    }
)

# Fit model
model.fit(
    x=[X_bin_train, X_multi_train, X_house_train, X_mnist[:10000]],  # sample MNIST for speed
    y=[y_bin_train, y_multi_train, y_house_train, y_mnist[:10000]],
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# Evaluate each task
results = model.evaluate(
    [X_bin_test, X_multi_test, X_house_test, X_mnist_test[:2000]],
    [y_bin_test, y_multi_test, y_house_test, y_mnist_test[:2000]],
    verbose=0
)
print("Evaluation (Binary, Multi, House, MNIST):", results)
