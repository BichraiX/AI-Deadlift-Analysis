#Complete Pipeline of the app

print("Loading libraries and models...")

import os                      
import helper_functions as hf   
import cv2                      
from ultralytics import YOLO  
import numpy as np              
import pandas as pd           

### Export joint positions and barbell bounding box data for all frames as csv

#Load the YOLO models
model_barbell = YOLO('models/best_barbell_detector_bar.pt')  # Our fine-tuned model for the detection of the barbell
model_pose = YOLO("models/yolo11x-pose.pt")  # Ultralytics' model for the pose detection

video_path = input("Video path: ")  

output_csv_pose = "joints_position.csv"

# Call the function to extract in a csv joint positions for all frames
print("Extracting joint positions...")
hf.extract_joint_positions(video_path, model_pose, output_csv_pose, debug=True)
print("Joint positions and barbell bounding box data exported successfully!")


###Check the orientation of the video (Sometimes does not work because nano model does not handle well the upside down videos)

print("Preparing the video for classification...")

#Load the CSV file
df_joints = pd.read_csv(output_csv_pose)

# Extract upper body and lower body mean positions
upper_body = ['nose_y', 'left_ear_y', 'right_ear_y', 'left_eye_y', 'right_eye_y', 'left_shoulder_y', 'right_shoulder_y']
upper_y = df_joints[upper_body].stack().mean()

lower_body = ['left_knee_y', 'right_knee_y', 'left_ankle_y', 'right_ankle_y']
lower_y = df_joints[lower_body].stack().mean()

# Calculate the difference between the head and knee positions
diff_up_low = upper_y - lower_y

# Check the orientation of the video
if np.mean(diff_up_low) < 0:
    vertical = True
else:
    vertical = False

print(upper_y)
print(lower_y)

print(vertical)

### Resize and rotate the video
prepocessed_video = "preprocessed.mp4"
res = (1280, 720)  # Output resolution 1280x720 (16:9 720p)
fps = 30 
hf.process_video(video_path, prepocessed_video, res, fps, vertical)

print("Video pre-processed successfully!")


output_csv_barbell = "barbell_positions.csv"

# Call the function to extract the barbell positions
hf.extract_barbell_positions(prepocessed_video, model_barbell, 'barbell_positions.csv', debug=True)

# Call the function to extract in a csv joint positions for all frames in the new video
output_csv_new_pose = "joints_position_new.csv"
print("Extracting joint positions...")
hf.extract_joint_positions(prepocessed_video, model_pose, output_csv_new_pose, debug=True)
print("Joint positions and barbell bounding box data exported successfully!")

df_joints_new = pd.read_csv(output_csv_new_pose)
df_barbell = pd.read_csv(output_csv_barbell)

# Calculate the mean y position of the barbell at each frame
barbell_y_means = df_barbell[['barbell_y_min', 'barbell_y_max']].mean(axis=1).to_numpy()
knee_y = df_joints_new[['left_knee_y', 'right_knee_y']].mean(axis=1).to_numpy()

# Get the phases of the lift

smoothing = 5 # Smoothing factor for the derivative
alpha = 0.1  # Weight for the derivative

phases = hf.get_phase(barbell_y_means, knee_y, smoothing, alpha) #to be implemented

# Output directory for the separated phases
output_dir = "phases"

# Separate the video into phases
hf.separate_phases(prepocessed_video, phases, output_dir)



# Get the list of video files in the output directory
video_files = [f for f in os.listdir(output_dir) if f.endswith(('.mp4'))]

results = {}

for video_file in video_files:
    part_of_movement_path = os.path.join(output_dir, video_file)
    print(f"Processing video: {video_file}")
    
    # Apply the estimate_movement function to the video
    try:
        video_path = f"phases/{video_file}"
        movement_result = hf.estimate_movement(video_path, model_pose, model_barbell) # Movement estimation with one model per phase
        results[video_file] = movement_result
        print(f"Result for {video_file}: {movement_result}")
    except Exception as e:
        print(f"Error processing {video_file}: {e}")
