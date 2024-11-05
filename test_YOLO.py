from ultralytics import YOLO
import pandas as pd

# Load the model
model = YOLO("models/yolo11n-pose.pt")

# List to store joint position data
all_joints = []

# Process the video to track keypoints
results = model.track(source="dataset/Deadlift.mp4", show=True, save=True)

# Define the expected joint labels
joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

# Loop through the results
for frame_id, result in enumerate(results):
    for person_id, person in enumerate(result.keypoints):  # assuming each `result` has a `keypoints` attribute
        keypoints = person.xy if hasattr(person, "xy") else []  # get keypoint coordinates or an empty list if missing

        # Create a dictionary for each person in each frame
        data = {
            "frame_id": frame_id,
            "person_id": person_id
        }

        # Loop through each joint label and assign coordinates if available
        for i, label in enumerate(joint_labels):
            if i < len(keypoints) and len(keypoints[i]) >= 2:  # Check if keypoint data is present and has x, y
                data[f"{label}_x"] = keypoints[i][0]  # x-coordinate of the joint
                data[f"{label}_y"] = keypoints[i][1]  # y-coordinate of the joint
            else:
                # Assign None if the keypoint data is missing
                data[f"{label}_x"] = None
                data[f"{label}_y"] = None

        # Append to the list of all joints
        all_joints.append(data)

# Convert to a DataFrame
joints_df = pd.DataFrame(all_joints)

# Save the DataFrame to a CSV file for further analysis
joints_df.to_csv("joint_positions_output.csv", index=False)

# Display DataFrame for inspection
joints_df.head()
