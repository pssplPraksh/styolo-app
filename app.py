# Import required libraries
import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import csv
import time
#Setting page layout
st.set_page_config(
    page_title="Video Object Detection",
    page_icon="📹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.sidebar.header("Model Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7)

# Main content area
st.title("Video Object Detection")

# File uploader for video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi"], key="video")
file_upload = False
# Process video and display results
if uploaded_file:
    # Read the video file using OpenCV
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(uploaded_file.read())
        video_path = temp_file.name
        file_upload = True
    if file_upload == True:
        #begin = time.time() 
        import add_missing_data as amd
        import visualize
        import main
        csv_path = 'test.csv'
        # Process the video
        main.main_process(video_path, csv_path)
        # Remove the temporary video file
        # Load the CSV file
        with open(csv_path, 'r') as file:
            reader = csv.DictReader(file)
            data = list(reader)

        # Interpolate missing data
        interpolated_data = amd.interpolate_bounding_boxes(data)
        csv_path2 = 'test_interpolated.csv'
        # Write updated data to a new CSV file
        header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 'license_plate_bbox_score', 'license_number', 'license_number_score']
        with open(csv_path2, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            writer.writerows(interpolated_data)
        
        visualize.vis(video_path, csv_path2)

    os.unlink(video_path)
file_upload = False
#print(time.time()-begin)
