from ultralytics import YOLO
import cv2
import numpy as np

model = YOLO("models/yolo11n-pose.pt")

joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

video_source = "dataset/example1_good.mp4"

def calculate_angle(p1, p2, p3):
    if None in (p1, p2, p3) or None in (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Phase Check Functions
def check_setup_phase(hip_angle, knee_angle):
    feedback = ""
    if not (50 <= hip_angle <= 60):
        feedback += f"Adjust hip angle in setup: {hip_angle:.2f}° (ideal: 50-60°)\n"
    if not (100 <= knee_angle <= 110):
        feedback += f"Adjust knee angle in setup: {knee_angle:.2f}° (ideal: 100-110°)\n"
    return feedback

def check_initial_pull_phase(hip_angle, knee_angle):
    feedback = ""
    if not (110 <= hip_angle <= 135):
        feedback += f"Adjust hip angle in initial pull: {hip_angle:.2f}° (ideal: 110-135°)\n"
    if not (110 <= knee_angle <= 135):
        feedback += f"Adjust knee angle in initial pull: {knee_angle:.2f}° (ideal: 110-135°)\n"
    return feedback

def check_mid_pull_phase(hip_angle, knee_angle):
    feedback = ""
    if not (80 <= hip_angle <= 85):
        feedback += f"Adjust hip angle in mid-pull: {hip_angle:.2f}° (ideal: 80-85°)\n"
    if not (160 <= knee_angle <= 170):
        feedback += f"Adjust knee angle in mid-pull: {knee_angle:.2f}° (ideal: 160-170°)\n"
    return feedback

def check_lockout_phase(hip_angle, knee_angle):
    feedback = ""
    if hip_angle < 160:
        feedback += f"Adjust hip angle in lockout: {hip_angle:.2f}° (should be fully extended)\n"
    if knee_angle < 170:
        feedback += f"Adjust knee angle in lockout: {knee_angle:.2f}° (should be fully extended)\n"
    return feedback

def check_descent_phase(hip_angle, knee_angle):
    feedback = ""
    if not (60 <= hip_angle <= 70):
        feedback += f"Adjust hip angle in descent: {hip_angle:.2f}° (ideal: 60-70°)\n"
    if not (100 <= knee_angle <= 110):
        feedback += f"Adjust knee angle in descent: {knee_angle:.2f}° (ideal: 100-110°)\n"
    return feedback

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
results = model.track(source=video_source, stream=True)

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
