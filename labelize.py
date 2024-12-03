from ultralytics import YOLO
import cv2
import numpy as np

pose_detection_model = YOLO("models/yolo11n-pose.pt")

joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

video_source = "dataset/example1_good.mp4"

 

# Main function to check deadlift form
def check_deadlift_form(joint_data):
    left_shoulder = joint_data.get("left_shoulder", None)
    left_hip = joint_data.get("left_hip", None)
    left_knee = joint_data.get("left_knee", None)
    left_ankle = joint_data.get("left_ankle", None)

    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # Check each phase based on hip and knee angles
    feedback = ""
    if hip_angle is not None and knee_angle is not None:
        feedback += check_setup_phase(hip_angle, knee_angle)
        feedback += check_initial_pull_phase(hip_angle, knee_angle)
        feedback += check_mid_pull_phase(hip_angle, knee_angle)
        feedback += check_lockout_phase(hip_angle, knee_angle)
        feedback += check_descent_phase(hip_angle, knee_angle)
    else:
        feedback += "Angle data unavailable for form check.\n"

    return feedback

# Process video and check form
results = pose_detection_model.track(source=video_source, stream=True)

for frame_id, result in enumerate(results):
    frame = result.orig_img

    sorted_people = sorted(result.keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True)
    
    if sorted_people:  
        person = sorted_people[0]
        keypoints = person.xy if hasattr(person, "xy") else []
        joint_data = {}
        
        for i, label in enumerate(joint_labels):
            x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
            joint_data[label] = (x, y)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        feedback_text = check_deadlift_form(joint_data)

        y_offset = 50  
        for line in feedback_text.splitlines():
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20 

    cv2.imshow("Deadlift Form Feedback", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
