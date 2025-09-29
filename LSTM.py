import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, GRU, LSTM, Dense, ConvLSTM2D, Flatten

# -----------------------------
# 1. SimpleRNN / GRU / LSTM Example (IMDB)
# -----------------------------
max_features = 10000
max_len = 200

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=max_features)
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

def train_sequence_rnn(rnn_type='SimpleRNN', n_nodes=32, epochs=2):
    model = Sequential()
    model.add(Embedding(max_features, 128, input_length=max_len))
    
    if rnn_type == 'SimpleRNN':
        model.add(SimpleRNN(n_nodes))
    elif rnn_type == 'GRU':
        model.add(GRU(n_nodes))
    elif rnn_type == 'LSTM':
        model.add(LSTM(n_nodes))
    else:
        raise ValueError("rnn_type must be 'SimpleRNN', 'GRU', or 'LSTM'")
    
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(f"\nTraining {rnn_type} model...")
    model.fit(X_train, y_train, epochs=epochs, batch_size=64, validation_split=0.2)
    loss, acc = model.evaluate(X_test, y_test)
    print(f"{rnn_type} Test Accuracy: {acc:.4f}")
    return model

# Train and compare
simple_rnn_model = train_sequence_rnn('SimpleRNN')
gru_model = train_sequence_rnn('GRU')
lstm_model = train_sequence_rnn('LSTM')

# -----------------------------
# 2. ConvLSTM2D Example (Simulated Video Data)
# -----------------------------
# Simulate video: 100 samples, 10 frames, 16x16 pixels, 1 channel, binary labels
num_samples = 100
frames = 10
height = 16
width = 16
channels = 1

X_video = np.random.rand(num_samples, frames, height, width, channels).astype(np.float32)
y_video = np.random.randint(0, 2, size=(num_samples, 1))

# Split train/test
split = int(0.8 * num_samples)
X_train_vid, X_test_vid = X_video[:split], X_video[split:]
y_train_vid, y_test_vid = y_video[:split], y_video[split:]

# Build ConvLSTM2D model
conv_lstm_model = Sequential()
conv_lstm_model.add(ConvLSTM2D(filters=16, kernel_size=(3,3), input_shape=(frames, height, width, channels),
                               activation='tanh', padding='same', return_sequences=False))
conv_lstm_model.add(Flatten())
conv_lstm_model.add(Dense(1, activation='sigmoid'))

conv_lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print("\nTraining ConvLSTM2D model...")
conv_lstm_model.fit(X_train_vid, y_train_vid, epochs=5, batch_size=8, validation_split=0.2)
loss, acc = conv_lstm_model.evaluate(X_test_vid, y_test_vid)
print(f"ConvLSTM2D Test Accuracy: {acc:.4f}")
