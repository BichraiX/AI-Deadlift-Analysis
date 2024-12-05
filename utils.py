## File containing auxiliary functions, namely those used to label the videos at 90 degrees points of view.
import numpy as np

def calculate_angle(p1, p2, p3):
    if None in (p1, p2, p3) or None in (p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]):
        return None
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    cosine_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

# Phase Check Functions
def check_setup_phase(hip_knee_ankle_angle, head_shoulder_hip_angle, shoulder_barbell_midfoot_angle):
    # Check for rounded upper back
    if head_shoulder_hip_angle < 170:
        return "Setup error: Rounded upper back (thoracic spine). Maintain a neutral spine."
    # Check for hips too low or too high
    if hip_knee_ankle_angle < 120 :
        return "Setup error: Hips are too low (like a squat)."
    if hip_knee_ankle_angle > 140:
        return "Setup error: Hips are too high (like a stiff-legged deadlift)."
    # Check for bar too far from the shins
    if shoulder_barbell_midfoot_angle > 90:
        return "Setup error: Barbell is too far from the shins. Keep it closer to the midfoot."
    return "Setup phase is correct."

def check_ascending_phase(hip_knee_shoulder_angle, barbell_midfoot_angle, lumbar_spine_angle):
    # Check for rounded lower back
    if lumbar_spine_angle < 170:
        return "Ascending error: Rounded lower back (lumbar spine). Maintain a neutral spine to avoid injury."
    # Check for hips rising too fast
    if hip_knee_shoulder_angle > 120:  # Angle increasing faster than expected
        return "Ascending error: Hips are rising too fast, causing a good morning effect."
    # Check for bar path deviation
    if barbell_midfoot_angle > 10:
        return "Ascending error: The barbell is drifting away from the body. Keep it close to your shins."
    return "Ascending phase is correct."

def check_lockout_phase(head_shoulder_hip_angle, hip_knee_ankle_angle, shoulder_barbell_hip_angle):
    # Check for overextension of the back
    if head_shoulder_hip_angle > 200:
        return "Lockout error: Overextension of the back (hyperextension). Keep a neutral spine at the top."
    # Check for underutilized glutes
    if hip_knee_ankle_angle < 170:
        return "Lockout error: Incomplete lockout due to insufficient glute engagement."
    # Check for shoulders behind the bar
    if shoulder_barbell_hip_angle < 90:
        return "Lockout error: Shoulders are drifting too far behind the bar. Align them vertically."
    return "Lockout phase is correct."

def check_descending_phase(shoulder_hip_knee_angle, hip_knee_ankle_angle, barbell_midfoot_angle, lumbar_spine_angle):
    # Check for rounded lower back
    if lumbar_spine_angle < 170:
        return "Descending error: Rounded lower back. Maintain a neutral spine during the descent."
    # Check for lack of hip hinge
    if shoulder_hip_knee_angle > hip_knee_ankle_angle:  # Hip angle increases before knee angle
        return "Descending error: Lack of proper hip hinge. Avoid squatting the descent."
    # Check for barbell drifting forward
    if barbell_midfoot_angle > 10:
        return "Descending error: Barbell is drifting away from the legs. Keep it close to your shins."
    return "Descending phase is correct."

def check_deadlift_form(joint_data, barbell_coords):
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
    hip_knee_ankle_angle = calculate_angle(hip, knee, ankle)
    head_shoulder_hip_angle = calculate_angle(ear, shoulder, hip)
    shoulder_barbell_midfoot_angle = calculate_angle(shoulder, barbell_coords, ankle)
    barbell_midfoot_angle = calculate_angle(barbell_coords, knee, ankle)
    hip_knee_shoulder_angle = calculate_angle(hip, knee, shoulder)
    lumbar_spine_angle = calculate_angle(hip, shoulder, ear)
    shoulder_barbell_hip_angle = calculate_angle(shoulder, barbell_coords, hip)
    shoulder_hip_knee_angle = calculate_angle(shoulder, hip, knee)

    ## add a condition to check that all of the angles are not None
    if None in (hip_knee_ankle_angle, head_shoulder_hip_angle, shoulder_barbell_midfoot_angle, barbell_midfoot_angle, hip_knee_shoulder_angle, lumbar_spine_angle, shoulder_barbell_hip_angle, shoulder_hip_knee_angle):
        return "Angle data unavailable for form check.\n"
    # Check each phase based on hip and knee angles
    else :
        feedback = ""
        feedback += check_setup_phase(hip_knee_ankle_angle, head_shoulder_hip_angle, shoulder_barbell_midfoot_angle)
        feedback += check_ascending_phase(hip_knee_shoulder_angle, barbell_midfoot_angle, lumbar_spine_angle)
        feedback += check_lockout_phase(head_shoulder_hip_angle, hip_knee_ankle_angle, shoulder_barbell_hip_angle)
        feedback += check_descending_phase(shoulder_hip_knee_angle, hip_knee_ankle_angle, barbell_midfoot_angle, lumbar_spine_angle)
    return feedback