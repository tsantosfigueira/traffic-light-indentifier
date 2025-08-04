import streamlit as st
import numpy as np
import cv2
from PIL import Image
from ultralytics import YOLO

model = YOLO('runs/detect/train/weights/best.pt')

st.title("Color and Traffic Light Detector (YOLOv8)")
st.write("Upload an image: ")

uploaded_file = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    results = model.predict(source=image_np, conf=0.25)

    annotated_frame = results[0].plot()

    st.image(annotated_frame, caption="Result", use_container_width=True)

    if len(results[0].boxes) > 0:
        st.subheader("Detected traffic lights:")
        for box in results[0].boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls]
            st.markdown(f"- **{label.capitalize()}** (confidence: {conf:.2f})")

    else:
        st.subheader("No traffic lights detected!")
