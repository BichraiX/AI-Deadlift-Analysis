import os
from ultralytics import YOLO
import cv2
import numpy as np
from utils import calculate_angle, check_lockout_phase, check_descending_phase, check_ascending_phase, check_setup_phase


pose_detection_model = YOLO("models/yolo11n-pose.pt")
barbell_detection_model = YOLO("models/best_barbell_detector_bar.pt")

joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

input_folder = "sep_alpha_0.1/sep_90/bad/"
output_folder = "processed_videos/"

# Create output folders if not exist
os.makedirs(output_folder, exist_ok=True)

def save_video(video_name, frames, output_path):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    for frame in frames:
        out.write(frame)
    out.release()

# Process videos in the folder
for video_file in os.listdir(input_folder):
    if not video_file.endswith(('.mp4', '.avi', '.mov')):
        continue

    video_path = os.path.join(input_folder, video_file)
    pose_results = pose_detection_model.track(source=video_path, stream=True)
    barbell_results = barbell_detection_model.track(source=video_path, stream=True)

    processed_frames = []
    phase_function = None

    # Determine which function to apply
    if "still_up" in video_file:
        phase_function = check_lockout_phase
    elif "up" in video_file and "still_up" not in video_file:
        phase_function = check_ascending_phase
    elif "still_down" in video_file:
        phase_function = check_setup_phase
    elif "down" in video_file and "still_down" not in video_file:
        phase_function = check_descending_phase

    for ((frame_id, pose_result), (_, barbell_result)) in zip(enumerate(pose_results), enumerate(barbell_results)):
        frame = pose_result.orig_img
        sorted_people = sorted(pose_result.keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True)

        if sorted_people:
            person = sorted_people[0]
            keypoints = person.xy if hasattr(person, "xy") else []
            joint_data = {}

            for i, label in enumerate(joint_labels):
                x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                joint_data[label] = (x, y)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

            # Calculate joint positions
            shoulder = (
                (joint_data['left_shoulder'][0] + joint_data['right_shoulder'][0]) / 2,
                (joint_data['left_shoulder'][1] + joint_data['right_shoulder'][1]) / 2
            )
            hip = (
                (joint_data['left_hip'][0] + joint_data['right_hip'][0]) / 2,
                (joint_data['left_hip'][1] + joint_data['right_hip'][1]) / 2
            )
            knee = (
                (joint_data['left_knee'][0] + joint_data['right_knee'][0]) / 2,
                (joint_data['left_knee'][1] + joint_data['right_knee'][1]) / 2
            )
            ankle = (
                (joint_data['left_ankle'][0] + joint_data['right_ankle'][0]) / 2,
                (joint_data['left_ankle'][1] + joint_data['right_ankle'][1]) / 2
            )
            ear = (
                (joint_data['left_ear'][0] + joint_data['right_ear'][0]) / 2,
                (joint_data['left_ear'][1] + joint_data['right_ear'][1]) / 2
            )

            # Calculate barbell position
            if barbell_result.boxes:
                largest_box = max(barbell_result.boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                barbell_coords = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            else:
                barbell_coords = None


            # Calculate angles and apply the appropriate function
            feedback_text = ""
            if barbell_coords is not None and phase_function == check_setup_phase:
                hip_knee_ankle_angle = calculate_angle(hip, knee, ankle)
                head_shoulder_hip_angle = calculate_angle(ear, shoulder, hip)
                shoulder_barbell_midfoot_angle = calculate_angle(shoulder, barbell_coords, ankle)
                feedback_text = phase_function(hip_knee_ankle_angle, head_shoulder_hip_angle, shoulder_barbell_midfoot_angle)
            elif phase_function == check_ascending_phase:
                hip_knee_shoulder_angle = calculate_angle(hip, knee, shoulder)
                barbell_midfoot_angle = calculate_angle(barbell_coords, knee, ankle)
                lumbar_spine_angle = calculate_angle(shoulder, hip, knee)
                feedback_text = phase_function(hip_knee_shoulder_angle, barbell_midfoot_angle, lumbar_spine_angle)
            elif phase_function == check_lockout_phase:
                head_shoulder_hip_angle = calculate_angle(ear, shoulder, hip)
                hip_knee_ankle_angle = calculate_angle(hip, knee, ankle)
                shoulder_barbell_hip_angle = calculate_angle(shoulder, barbell_coords, hip)
                feedback_text = phase_function(head_shoulder_hip_angle, hip_knee_ankle_angle, shoulder_barbell_hip_angle)
            elif phase_function == check_descending_phase:
                shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)
                hip_knee_ankle_angle = calculate_angle(hip, knee, ankle)
                barbell_midfoot_angle = calculate_angle(barbell_coords, knee, ankle)
                lumbar_spine_angle = calculate_angle(shoulder, hip, knee)
                feedback_text = phase_function(shoulder_hip_knee_angle, hip_knee_ankle_angle, barbell_midfoot_angle, lumbar_spine_angle)

            # Display feedback
            y_offset = 50
            for line in feedback_text.splitlines():
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                y_offset += 20

        processed_frames.append(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    # Save processed video
    if processed_frames:
        output_subfolder = os.path.join(output_folder, feedback_text)
        os.makedirs(output_subfolder, exist_ok=True)
        output_path = os.path.join(output_subfolder, video_file)
        save_video(video_file, processed_frames, output_path)
