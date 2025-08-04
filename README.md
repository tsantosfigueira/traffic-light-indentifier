# Traffic Light Identifier with YOLOv8

This project aims to detect traffic lights and classify their current color (red, yellow, or green) using a YOLOv8 object detection model.

---

## Project Overview

The system detects traffic lights in images and classifies their color. The solution is based on the Ultralytics implementation of YOLOv8.

---

## Step-by-Step Pipeline

### 1. Dataset Preparation

- A public dataset of traffic lights was used and processed using [Roboflow](https://app.roboflow.com/trafficlight-kwocw/capstone-2ax1t-fo6hi/1).
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
- The application uses YOLOv8m (medium) — a balanced model from the YOLOv8 family that offers a good trade-off between speed and accuracy. Compared to smaller variants like yolov8n.pt or yolov8s.pt, yolov8m.pt provides higher detection precision, making it more suitable for tasks where reliability is important, such as traffic light classification. It is optimized for use on modern GPUs and performs well in real-time scenarios.

### 4. Model training
- The training loss and evaluation metrics were monitored over 50 epochs.
- Outputs and logs were saved in the runs/detect/train directory.
- The training script it Main.py

### 5. Model evaluation
- mAP@0.5: Measures average precision at IOU ≥ 0.5
- mAP@0.5:0.95: COCO-style stricter evaluation
- Precision and Recall: to measure false positives and missed detections

The final model achieved a mAP@0.5 of ~0.51 and mAP@0.5:0.95 of ~0.25.


## Traffic Light Application

<img width="812" height="883" alt="image" src="https://github.com/user-attachments/assets/5043ffec-16fc-4787-a325-3ceb6c3ef61d" />

The project includes a simple web interface built with Streamlit that allows users to:

📤 Upload an image

🤖 Automatically detect traffic lights in the image using a YOLOv8 model

🎯 Identify the current color of each traffic light (red, yellow, or green)

🖼️ Visualize the output image with annotated bounding boxes and labels

The application runs locally and performs inference using the trained model (best.pt) with no need for manual image preprocessing.

