import os
import cv2
import numpy as np
import pandas as pd
import helper_functions as hf
from ultralytics import YOLO

# Load YOLO models
model_barbell = YOLO('models/best_barbell_detector_bar.pt')  # Barbell detection model
model_pose = YOLO("models/yolo11x-pose.pt")  # Pose detection model

# Directories
input_dirs = {"good": "dataset/good/", "bad": "dataset/bad/"}
output_dirs = {"good": "dataset/good_separated/", "bad": "dataset/bad_separated/"}

# Ensure output directories exist
for output_dir in output_dirs.values():
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

# Process each dataset (good and bad)
for category, input_dir in input_dirs.items():
    output_dir = output_dirs[category]
    print(f"Processing {category} videos...")

    # Iterate over all videos in the folder
    for video_name in os.listdir(input_dir):
        video_path = os.path.join(input_dir, video_name)

        # Skip non-video files
        if not video_name.lower().endswith(('.mp4', '.avi', '.mov')):
            print(f"Skipping non-video file: {video_name}")
            continue

        print(f"Processing video: {video_name}")

        # Prepare paths for CSV files
        base_name = os.path.splitext(video_name)[0]
        joint_csv_path = os.path.join(output_dir, f"{base_name}_joints.csv")
        barbell_csv_path = os.path.join(output_dir, f"{base_name}_barbell.csv")
        joint_new_csv_path = os.path.join(output_dir, f"{base_name}_joints_new.csv")

        preprocessed_video_path = video_path


        """
        # Extract joint positions
        hf.extract_joint_positions(video_path, model_pose, joint_csv_path, debug=False)

        # Load joint data
        df_joints = pd.read_csv(joint_csv_path)

        # Check the orientation of the video
        upper_body = ['nose_y', 'left_ear_y', 'right_ear_y', 'left_eye_y', 'right_eye_y', 'left_shoulder_y', 'right_shoulder_y']
        upper_y = df_joints[upper_body].stack().mean()

        lower_body = ['left_knee_y', 'right_knee_y', 'left_ankle_y', 'right_ankle_y']
        lower_y = df_joints[lower_body].stack().mean()

        vertical = upper_y < lower_y

        # Preprocess the video
        preprocessed_video_path = os.path.join(output_dir, f"{base_name}_preprocessed.mp4")
        hf.process_video(video_path, preprocessed_video_path, (1280, 720), 30, vertical)

        """

        # Extract barbell positions
        hf.extract_barbell_positions(preprocessed_video_path, model_barbell, barbell_csv_path, debug=False)

        # Extract joint positions again for the preprocessed video
        hf.extract_joint_positions(preprocessed_video_path, model_pose, joint_new_csv_path, debug=False)

        # Load barbell and joint data
        df_barbell = pd.read_csv(barbell_csv_path)
        df_joints_new = pd.read_csv(joint_new_csv_path)

        # Calculate barbell and knee positions
        barbell_y_means = df_barbell[['barbell_y_min', 'barbell_y_max']].mean(axis=1).to_numpy()
        knee_y = df_joints_new[['left_knee_y', 'right_knee_y']].mean(axis=1).to_numpy()

        # Get phases
        smoothing = 5  # Smoothing factor for the derivative
        alpha = 0.05  # Weight for the derivative
        N = 3 # Smoothing factor for the phase detection to avoid flickering over one frame
        phases = hf.get_phase(barbell_y_means, knee_y, smoothing, alpha, N)

        # Separate the video into phases
        hf.separate_phases(preprocessed_video_path, phases, output_dir)

        print(f"Video {video_name} processed and saved to {output_dir} with CSVs")

print("All videos processed successfully!")
