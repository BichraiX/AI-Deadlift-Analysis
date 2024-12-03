## File containing auxiliary functions, namely those used to label the videos at 90 degrees points of view.

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
    if hip_knee_ankle_angle < 120 or hip_knee_ankle_angle > 140:
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

def check_lockout_phase(head_shoulder_hip_angle, hip_knee_angle, shoulder_barbell_hip_angle):
    # Check for overextension of the back
    if head_shoulder_hip_angle > 200:
        return "Lockout error: Overextension of the back (hyperextension). Keep a neutral spine at the top."
    # Check for underutilized glutes
    if hip_knee_angle < 170:
        return "Lockout error: Incomplete lockout due to insufficient glute engagement."
    # Check for shoulders behind the bar
    if shoulder_barbell_hip_angle > 90:
        return "Lockout error: Shoulders are drifting too far behind the bar. Align them vertically."
    return "Lockout phase is correct."

def check_descending_phase(hip_shoulder_angle, knee_angle, barbell_midfoot_angle, lumbar_spine_angle):
    # Check for rounded lower back
    if lumbar_spine_angle < 170:
        return "Descending error: Rounded lower back. Maintain a neutral spine during the descent."
    # Check for lack of hip hinge
    if hip_shoulder_angle > knee_angle:  # Hip angle increases before knee angle
        return "Descending error: Lack of proper hip hinge. Avoid squatting the descent."
    # Check for barbell drifting forward
    if barbell_midfoot_angle > 10:
        return "Descending error: Barbell is drifting away from the legs. Keep it close to your shins."
    return "Descending phase is correct."
