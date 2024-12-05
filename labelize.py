from ultralytics import YOLO
import cv2
import numpy as np
from utils import check_deadlift_form

pose_detection_model = YOLO("models/yolo11n-pose.pt")
barbell_detection_model = YOLO("models/best_barbell_detector_bar.pt")

joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

video_source = "dataset/good/good001.mp4"

# Process video and check form
pose_results = pose_detection_model.track(source=video_source, stream=True)
barbell_results = barbell_detection_model.track(source=video_source, stream=True)

for ((frame_id, pose_result),(_, barbell_result)) in zip(enumerate(pose_results),enumerate(barbell_results)):
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
        
        largest_box = max(barbell_result.boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
        x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
        barbell_coords = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        feedback_text = check_deadlift_form(joint_data, barbell_coords)
        y_offset = 50  
        for line in feedback_text.splitlines():
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20 

    cv2.imshow("Deadlift Form Feedback", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
