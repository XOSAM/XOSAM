import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications import ResNet50, VGG16

print("Available GPUs:", tf.config.list_physical_devices('GPU'))

# -----------------------------
# Parameters
# -----------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 30

TRAIN_IMAGES_PATH = "train/images/"
TRAIN_MASKS_PATH = "train/masks/"
TEST_IMAGES_PATH = "test/images/"

# -----------------------------
# Data Loading
# -----------------------------
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
X_train, X_val = train_images[:int(0.9*len(train_images))], train_images[int(0.9*len(train_images)):]
y_train, y_val = train_masks[:int(0.9*len(train_masks))], train_masks[int(0.9*len(train_masks)):]

test_images, test_img_files = load_images(TEST_IMAGES_PATH, size=(IMAGE_SIZE, IMAGE_SIZE))

# -----------------------------
# IoU Metric
# -----------------------------
def iou_metric(y_true, y_pred):
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred) - intersection
    return intersection / (union + 1e-6)

# -----------------------------
# U-Net with Pretrained Encoder
# -----------------------------
def build_unet_encoder(encoder_name='resnet', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)):
    inputs = Input(shape=input_shape)
    
    if encoder_name == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)
        skips = [
            base_model.get_layer("input_1").output,
            base_model.get_layer("conv1_relu").output,
            base_model.get_layer("conv2_block3_out").output,
            base_model.get_layer("conv3_block4_out").output,
            base_model.get_layer("conv4_block6_out").output
        ]
        x = base_model.get_layer("conv5_block3_out").output
    elif encoder_name == 'vgg':
        base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
        skips = [
            base_model.get_layer("block1_conv2").output,
            base_model.get_layer("block2_conv2").output,
            base_model.get_layer("block3_conv3").output,
            base_model.get_layer("block4_conv3").output
        ]
        x = base_model.get_layer("block5_conv3").output
    else:
        raise ValueError("Encoder must be 'resnet' or 'vgg'")

    # Decoder
    skips = skips[::-1]  # reverse order
    for i, skip in enumerate(skips):
        x = UpSampling2D((2,2))(x)
        x = concatenate([x, skip])
        x = Conv2D(64//(2**i), (3,3), activation='relu', padding='same')(x)
        x = Conv2D(64//(2**i), (3,3), activation='relu', padding='same')(x)
    
    outputs = Conv2D(1, (1,1), activation='sigmoid')(x)
    model = Model(inputs, outputs)
    return model

# -----------------------------
# Train and Evaluate
# -----------------------------
def train_model(encoder_name):
    print(f"Training {encoder_name}-U-Net...")
    model = build_unet_encoder(encoder_name=encoder_name, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    model.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=[iou_metric])
    model.summary()
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
    model.save(f"{encoder_name}_unet_model.h5")
    
    preds = model.predict(test_images)
    os.makedirs(f"predicted_masks_{encoder_name}", exist_ok=True)
    for i, pred in enumerate(preds):
        mask = (pred[:,:,0] > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(f"predicted_masks_{encoder_name}", test_img_files[i]), mask)
    return preds

# -----------------------------
# Run Training for ResNet and VGG
# -----------------------------
resnet_preds = train_model('resnet')
vgg_preds = train_model('vgg')

# -----------------------------
# Compare IoU on Validation Set
# -----------------------------
resnet_val_iou = iou_metric(tf.convert_to_tensor(y_val), tf.convert_to_tensor(resnet_preds[:len(y_val)]))
vgg_val_iou = iou_metric(tf.convert_to_tensor(y_val), tf.convert_to_tensor(vgg_preds[:len(y_val)]))
print("ResNet-U-Net IoU:", float(resnet_val_iou))
print("VGG-U-Net IoU:", float(vgg_val_iou))

# -----------------------------
# Visualize some predictions
# -----------------------------
for i in range(3):
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(test_images[i].squeeze(), cmap='gray')
    plt.title('Input Image')
    plt.subplot(1,3,2)
    plt.imshow(resnet_preds[i].squeeze(), cmap='gray')
    plt.title('ResNet Prediction')
    plt.subplot(1,3,3)
    plt.imshow(vgg_preds[i].squeeze(), cmap='gray')
    plt.title('VGG Prediction')
    plt.show()
