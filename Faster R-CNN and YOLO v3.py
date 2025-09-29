# Faster R-CNN and YOLOv3 batch inference
import os
import cv2
import numpy as np
from keras_frcnn import config, roi_helpers
from keras_frcnn import resnet as nn  # or import vgg as nn
from keras.models import load_model
from yolo import YOLO
from PIL import Image, ImageDraw

# ------------------- Faster R-CNN Inference -------------------
def run_faster_rcnn(image_path, output_path='output/frcnn_output.jpg'):
    C = config.Config()
    C.network = 'resnet50'
    C.model_path = 'model_frcnn.hdf5'
    C.num_rois = 32
    C.base_net_weights = nn.get_weight_path()

    model_rpn, model_classifier = nn.get_model(C)

    img = cv2.imread(image_path)
    orig_img = img.copy()
    X = np.expand_dims(cv2.resize(img, (600, 600)), axis=0)

    Y1, Y2, F = model_rpn.predict(X)
    R = roi_helpers.rpn_to_roi(Y1, Y2, C, K.image_data_format(), overlap_thresh=0.7)
    R[:, 2] -= R[:, 0]
    R[:, 3] -= R[:, 1]

    for jk in range(R.shape[0]//C.num_rois + 1):
        ROIs = R[C.num_rois*jk:C.num_rois*(jk+1), :]
        if ROIs.shape[0] == 0:
            break
        P_cls, P_regr = model_classifier.predict([F, ROIs])
        for ii in range(P_cls.shape[0]):
            cls_id = np.argmax(P_cls[ii])
            if cls_id != 0:
                x, y, w, h = ROIs[ii].astype(int)
                cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0,255,0), 2)
                cv2.putText(orig_img, str(cls_id), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(output_path, orig_img)
    print(f"Faster R-CNN output saved to {output_path}")

# ------------------- YOLOv3 Inference -------------------
def run_yolov3(image_path, output_path='output/yolo_output.jpg'):
    yolo_model = YOLO(
        **{
            "model_path": "yolo_weights.h5",
            "anchors_path": "yolo_anchors.txt",
            "classes_path": "coco_classes.txt",
            "score": 0.3,
            "iou": 0.45,
            "model_image_size": (416, 416)
        }
    )
    img = Image.open(image_path)
    boxes, scores, classes = yolo_model.detect_image(img)

    draw = ImageDraw.Draw(img)
    for b, c in zip(boxes, classes):
        top, left, bottom, right = b
        draw.rectangle([left, top, right, bottom], outline='red', width=2)
        draw.text((left, top-10), str(c), fill='red')

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    img.save(output_path)
    print(f"YOLOv3 output saved to {output_path}")

# ------------------- Batch Processing -------------------
def batch_process(input_folder='images', output_folder='output'):
    os.makedirs(output_folder, exist_ok=True)
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg','.png','.jpeg'))]

    for img_file in images:
        input_path = os.path.join(input_folder, img_file)
        frcnn_output = os.path.join(output_folder, 'frcnn_'+img_file)
        yolo_output = os.path.join(output_folder, 'yolo_'+img_file)

        print(f"Processing {img_file}...")
        run_faster_rcnn(input_path, frcnn_output)
        run_yolov3(input_path, yolo_output)

# ------------------- Example Usage -------------------
if __name__ == "__main__":
    batch_process(input_folder='images', output_folder='output')
                               
