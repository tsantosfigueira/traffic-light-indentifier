# Traffic Light Identifier with YOLOv8

This project aims to detect traffic lights and classify their current color (red, yellow, or green) using a YOLOv8 object detection model.

---

## Project Overview

The system detects traffic lights in images and classifies their color. The solution is based on the Ultralytics implementation of YOLOv8.

---

## Step-by-Step Pipeline

### 1. Dataset Preparation

- A public dataset of traffic lights was used and processed using [Roboflow](https://app.roboflow.com/trafficlight-kwocw/traffic-light-detection-bstld-ifr7x/1).
- Dataset was annotated with classes for each light color: `red`, `yellow`, and `green`.
- Preprocessing and augmentation were performed in Roboflow:
  - Resizing to 640x640
  - Flipping, brightness adjustment, zoom
- Dataset was exported in **YOLOv8 format**, including:
  - `/train/images`, `/train/labels`
  - `/valid/images`, `/valid/labels`
  - `data.yaml` (defining dataset structure and class names)

### 2. Environment Setup

A Python virtual environment was created using `venv`, and the required packages were installed:

```bash
python -m venv yolov8-env
yolov8-env\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python
```

The setup was verified on Python 3.10 with a NVIDIA RTX 4060 Ti and CUDA 12.1 support.

### 3. Model Selection
- The YOLOv8n (nano) model was selected for training due to its light weight and real-time inference capability.
- Pretrained weights from Ultralytics were used: yolov8n.pt.

### 4. Model training
- The training loss and evaluation metrics were monitored over 50 epochs.
- Outputs and logs were saved in the runs/detect/train directory.
- The training script it Main.py

### 5. Model evaluation
- mAP@0.5: Measures average precision at IOU ≥ 0.5
- mAP@0.5:0.95: COCO-style stricter evaluation
- Precision and Recall: to measure false positives and missed detections

The final model achieved a mAP@0.5 of ~0.51 and mAP@0.5:0.95 of ~0.25.

