import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import numpy as np

# Setting page layout
st.set_page_config(
    page_title="Video Object Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

vehicle_detector = YOLO('yolov8n.pt')
number_plate_detector = YOLO('final_number_plate_model.pt')

# Sidebar for model configuration
st.sidebar.header("Model Configuration")
confidence_threshold1 = st.sidebar.slider("Confidence Threshold car", 0.0, 1.0, 0.7)
confidence_threshold2 = st.sidebar.slider("Confidence Threshold license plate", 0.0, 1.0, 0.6)

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
        output_video_path = 'output.mp4'
        output_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'DIVX'), 30, (
            int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        # Process each frame of the video
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break

            # Perform object detection for vehicles
            vehicle_boxes = vehicle_detector.predict(frame, conf=confidence_threshold1)

            # Process vehicle boxes
            for box in vehicle_boxes[0].boxes:
                class_id = vehicle_boxes[0].names[box.cls[0].item()]
                cords = box.xywh[0].tolist()
                cords = [round(x) for x in cords]
                x, y, w, h = cords

                if class_id == 'car':
                    vehicle_image = frame[y:y + h, x:x + w]

                    # Detect number plate
                    plate_boxes = number_plate_detector.predict(vehicle_image, conf=confidence_threshold2)

                    # Process number plate boxes
                    for plate_box in plate_boxes[0].boxes:
                        cords2 = plate_box.xyxy[0].tolist()
                        cords2 = [round(j) for j in cords2]
                        cv2.rectangle(frame, (x + cords2[0], y + cords2[1]), (x + cords2[2], y + cords2[3]),
                                      (0, 255, 0), 2)

            # Write the processed frame to the output video
            output_video.write(frame)

        # Release video capture and writer objects
        video_capture.release()
        output_video.release()

        st.write("Processing completed!")
        st.write("Detection results saved.")

        # Convert the output video to binary data
        with open(output_video_path, 'rb') as file:
            video_binary = file.read()

        # Display a download button for the processed video
        st.download_button(label="Download Processed Video",
                           data=video_binary,
                           file_name="output.mp4",
                           mime="video/mp4")

        # Remove the temporary video file
        os.unlink(video_path)
        os.unlink(output_video_path)
    else:
        st.error("Error reading video file.")
