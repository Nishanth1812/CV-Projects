import os
import subprocess
import sys

# List of required packages
required_packages = [
    "streamlit",
    "ultralytics==8.1.0",
    "opencv-python",
    "pillow",
    "numpy",
    "requests",
    "torch",
    "torchvision"
]

# Install missing packages
for package in required_packages:
    try:
        __import__(package.split("==")[0])  # Import only the base module name
    except ModuleNotFoundError:
        print(f"{package} not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Now import the installed packages
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import requests
from ultralytics import YOLO

# GitHub raw file URL for 'best.pt'
MODEL_URL = "https://github.com/Nishanth1812/YOLO-Streamlit-App/raw/main/best.pt"
MODEL_PATH = "best.pt"

# Download the model if it doesn't exist
if not os.path.exists(MODEL_PATH):
    st.info("Downloading YOLO model... This may take a few seconds.")
    response = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    st.success("Model downloaded successfully!")

# Load YOLO model
model = YOLO(MODEL_PATH)

# Streamlit UI
st.title("YOLO Object Detection App (Webcam)")

# Brightness slider
brightness = st.slider("Adjust Brightness", min_value=0.5, max_value=2.0, value=1.0, step=0.1)

start_button = st.button("Start Webcam")
stop_button = st.button("Stop Webcam")

# Initialize webcam
if start_button:
    web_cam = cv2.VideoCapture(0)
    web_cam.set(3, 640)
    web_cam.set(4, 480)

    frame_placeholder = st.empty()

    while web_cam.isOpened():
        success, img_frame = web_cam.read()
        if not success:
            break  # Exit if frame not captured properly

        # Adjust brightness
        img_frame = cv2.convertScaleAbs(img_frame, alpha=brightness, beta=0)

        # Run YOLO model on the frame
        results = model(img_frame, conf=0.8)
        a_frame = results[0].plot()

        # Convert frame to RGB for Streamlit
        output_frame_rgb = cv2.cvtColor(a_frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(output_frame_rgb)

        # Display the frame in Streamlit
        frame_placeholder.image(img_pil, caption="Webcam Feed", use_column_width=True)

        if stop_button:
            break

    web_cam.release()
    cv2.destroyAllWindows()
