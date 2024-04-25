# Import required libraries
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Setting page layout
st.set_page_config(
    page_title="Video Object Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Model initialization pssplpraksh/styolo-app/main/app.py
model_path = r'pssplpraksh/styolo-app/main/final_number_plate_model.pt'
model = YOLO(model_path)

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# Main content area
st.title("Video Object Detection")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"], key="video")

# Process video and display results
if uploaded_file:
    # Read the video file using OpenCV
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name

    # Process the video
    video_capture = cv2.VideoCapture(video_path)

    if video_capture.isOpened():
        # Initialize video writer to save the output
        output_video = cv2.VideoWriter(r'C:\Users\Surojit\Downloads\output.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (
        int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Process each frame of the video
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Perform object detection
            res = model.predict(source=frame, conf=confidence_threshold)
            boxes = res[0].boxes

            # Draw bounding boxes on the frame
            for box in boxes:
                if len(box.xywh) == 4:
                    x, y, w, h = box.xywh
                    x1, y1, x2, y2 = int(x - w / 2), int(y - h / 2), int(x + w / 2), int(y + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Write the processed frame to the output video
            output_video.write(frame)

        # Release video capture and writer objects
        video_capture.release()
        output_video.release()

        st.write("Processing completed!")
        st.write("Detection results saved.")
    else:
        st.error("Error reading video file.")

    # Remove the temporary video file
    os.unlink(video_path)
