from ultralytics import YOLO
import cv2
import numpy as np

# Load the model
model = YOLO("models/yolo11n-pose.pt")

# Define joint labels and the video source
joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist", "left_hip", "right_hip",
                "left_knee", "right_knee", "left_ankle", "right_ankle"]

video_source = "dataset/example2.mp4"

# Function to calculate angles between three points
def calculate_angle(p1, p2, p3):
    if None in (p1, p2, p3) or None in (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Function to analyze deadlift form and provide feedback
def check_deadlift_form(joint_data):
    # Extract coordinates for relevant joints
    left_shoulder = joint_data.get("left_shoulder", None)
    left_hip = joint_data.get("left_hip", None)
    left_knee = joint_data.get("left_knee", None)
    left_ankle = joint_data.get("left_ankle", None)
    
    # Calculate angles
    hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
    
    # Generate feedback only if angles are valid
    feedback = ""
    if hip_angle is not None:
        if 30 <= hip_angle <= 45:
            feedback += "Good hip angle for starting position.\n"
        elif hip_angle > 160:
            feedback += "Good hip angle for lockout.\n"
        else:
            feedback += f"Adjust hip angle: {hip_angle:.2f}°\n"
    else:
        feedback += "Hip angle data unavailable.\n"
    
    if knee_angle is not None:
        if 70 <= knee_angle <= 90:
            feedback += "Good knee angle for starting position.\n"
        elif knee_angle > 160:
            feedback += "Good knee angle for lockout.\n"
        else:
            feedback += f"Adjust knee angle: {knee_angle:.2f}°\n"
    else:
        feedback += "Knee angle data unavailable.\n"
    
    return feedback

# Process the video and display form feedback
results = model.track(source=video_source, stream=True)

for frame_id, result in enumerate(results):
    frame = result.orig_img  # Original frame

    # Sort people by distance (assuming bounding box area indicates depth)
    sorted_people = sorted(result.keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True)
    
    if sorted_people:  # Check if any person is detected
        # Process only the closest person
        person = sorted_people[0]
        keypoints = person.xy if hasattr(person, "xy") else []
        joint_data = {}
        
        # Extract keypoints into the joint_data dictionary
        for i, label in enumerate(joint_labels):
            x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
            joint_data[label] = (x, y)
            print(x,y)
            # Draw the keypoint on the frame
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        
        # Get feedback for this person
        feedback_text = check_deadlift_form(joint_data)

        # Display feedback on the frame
        y_offset = 50  # Starting y position for text display
        for line in feedback_text.splitlines():
            cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            y_offset += 20  # Line spacing

    # Show the frame with feedback and keypoints
    cv2.imshow("Deadlift Form Feedback", frame)
    
    # Press 'q' to exit early
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cv2.destroyAllWindows()
