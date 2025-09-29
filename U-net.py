import os
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array

print("Available GPUs:", tf.config.list_physical_devices('GPU'))

IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50

TRAIN_IMAGES_PATH = "train/images/"
TRAIN_MASKS_PATH = "train/masks/"
TEST_IMAGES_PATH = "test/images/"

def load_images(path, size=(128,128), mask=False):
    imgs = []
    file_list = sorted(os.listdir(path))
    for f in tqdm(file_list):
        img = load_img(os.path.join(path, f), target_size=size, color_mode='grayscale')
        img = img_to_array(img)/255.0
        if mask:
            img = (img > 0).astype(np.float32)
        imgs.append(img)
    return np.array(imgs), file_list

train_images, train_img_files = load_images(TRAIN_IMAGES_PATH, size=(IMAGE_SIZE, IMAGE_SIZE))
train_masks, _ = load_images(TRAIN_MASKS_PATH, size=(IMAGE_SIZE, IMAGE_SIZE), mask=True)

X_train, X_val, y_train, y_val = train_test_split(train_images, train_masks, test_size=0.1, random_state=42)

def unet_model(input_size=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = Input(input_size)
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, (3,3), activation='relu', padding='same')(c1)
    p1 = MaxPooling2D((2,2))(c1)
    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(p1)
    c2 = Conv2D(32, (3,3), activation='relu', padding='same')(c2)
    p2 = MaxPooling2D((2,2))(c2)
    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = Conv2D(64, (3,3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2,2))(c3)
    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(p3)
    c4 = Conv2D(128, (3,3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2,2))(c4)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(p4)
    c5 = Conv2D(256, (3,3), activation='relu', padding='same')(c5)
    u6 = UpSampling2D((2,2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(u6)
    c6 = Conv2D(128, (3,3), activation='relu', padding='same')(c6)
    u7 = UpSampling2D((2,2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(u7)
    c7 = Conv2D(64, (3,3), activation='relu', padding='same')(c7)
    u8 = UpSampling2D((2,2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(u8)
    c8 = Conv2D(32, (3,3), activation='relu', padding='same')(c8)
    u9 = UpSampling2D((2,2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(u9)
    c9 = Conv2D(16, (3,3), activation='relu', padding='same')(c9)
    outputs = Conv2D(1, (1,1), activation='sigmoid')(c9)
    return Model(inputs=[inputs], outputs=[outputs])

model = unet_model()
model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
model.save("unet_salt_model.h5")

test_images, test_img_files = load_images(TEST_IMAGES_PATH, size=(IMAGE_SIZE, IMAGE_SIZE))
predictions = model.predict(test_images)

os.makedirs("predicted_masks", exist_ok=True)
for i, pred in enumerate(predictions):
    mask = (pred[:,:,0] > 0.5).astype(np.uint8) * 255
    cv2.imwrite(os.path.join("predicted_masks", test_img_files[i]), mask)

for i in range(3):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.subplot(1,2,2)
    plt.imshow(predictions[i].squeeze(), cmap='gray')
    plt.title('Predicted Mask')
    plt.show()
  
