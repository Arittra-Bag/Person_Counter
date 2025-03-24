import streamlit as st
import cv2
import tempfile
import time
import threading
import numpy as np
from inference_sdk import InferenceHTTPClient
from concurrent.futures import ThreadPoolExecutor
import os
import warnings

# Suppress ScriptRunContext warnings from threads
warnings.filterwarnings("ignore", message=".*missing ScriptRunContext.*")

# Global client using the inference SDK (for person counting)
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="clkjEzWr0PTQR8J0SgjV"
)

# Global variables for person counter mode
detections = []
num_people = 0

# Main app
st.title("👥 Person Counter")

# Person Counter specific settings
st.sidebar.header("Settings")
source = st.sidebar.radio("Input Source:", ["Webcam", "Upload Video"])
process_every_n_frames = st.sidebar.slider(
    "Process every N frames", 
    min_value=1, 
    max_value=30, 
    value=5,
    help="Higher values improve speed but may reduce accuracy"
)

# Thread-safe inference function using disk I/O (no in-memory images)
def run_inference(frame):
    global detections, num_people
    # Write the frame to disk (overwriting the same file each time)
    temp_filename = "temp_frame.jpg"
    cv2.imwrite(temp_filename, frame)
    try:
        # Use the crowd-density model for population density
        result = CLIENT.infer(temp_filename, model_id="crowd-density-ou3ne/1")
        detections = result.get('predictions', [])
        num_people = len(detections)
    except Exception as e:
        st.sidebar.error(f"Inference error: {str(e)}")
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

# Initialize thread pool for inference
executor = ThreadPoolExecutor(max_workers=1)
future = None

# Create a placeholder for the video display
frame_placeholder = st.empty()

# Process video frames
def process_video(video_source):
    global future
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        st.error("🚨 Error: Could not open video source!")
        return

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video

        frame_count += 1

        # Start inference on every Nth frame in a separate thread
        if frame_count % process_every_n_frames == 0:
            if future is None or future.done():
                future = executor.submit(run_inference, frame.copy())
        
        # Draw bounding boxes from the latest inference results
        for detection in detections:
            x, y, w, h = int(detection['x']), int(detection['y']), int(detection['width']), int(detection['height'])
            x1, y1 = x - w // 2, y - h // 2
            x2, y2 = x + w // 2, y + h // 2
            cv2.rectangle(frame, (x1, y1), (x2, y2), (170, 100, 200), 2)
            label = f"{detection.get('class', 'person')} {detection.get('confidence', 0):.2f}"
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (170, 100, 200), 2)

        # Display head count on the frame
        cv2.putText(frame, f"Total People: {num_people}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # Convert BGR to RGB for display in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
        
        time.sleep(0.1)  # Slight delay to reduce processing load

    cap.release()

# Handle video source selection
if source == "Webcam":
    st.sidebar.warning("🔴 Press Stop to end the stream")
    process_video(0)
elif source == "Upload Video":
    uploaded_file = st.sidebar.file_uploader(
        "📤 Upload Video File",
        type=["mp4", "avi", "mov", "mkv"],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        if st.sidebar.button("Start Processing"):
            process_video(tfile.name)
            tfile.close() 