import pandas as pd
import cv2
import numpy as np
import os
import contextlib
import sys
from classifiers_training.main_model import DeadliftMovementClassifier
import torch
import torch.nn.functional as F

def extract_joint_positions(video_path, model_pose, output_csv="joint_positions.csv", debug=False):
    """
    Extracts joint positions from a video and saves them to a CSV file. Optionally displays positions for debugging.

    Parameters:
        video_path (str): Path to the input video.
        model_pose: The pose estimation model.
        output_csv (str): Path to the output CSV file.
        debug (bool): If True, displays the video with overlaid joint positions.

    Outputs:
        Saves a CSV file with joint positions.
    """
    import cv2
    import pandas as pd

    joint_labels = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
                    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                    "left_wrist", "right_wrist", "left_hip", "right_hip",
                    "left_knee", "right_knee", "left_ankle", "right_ankle"]

    pose_estim = model_pose.track(source=video_path, stream=True)
    cap = cv2.VideoCapture(video_path) if debug else None  

    joint_positions = []

    for frame_id, pose in enumerate(pose_estim):
        frame_joint_positions = {"frame_id": frame_id}

        frame = cap.read()[1] if debug else None

        sorted_people = sorted(pose.keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True)
        if sorted_people:
            person = sorted_people[0]
            keypoints = person.xy if hasattr(person, "xy") else None

            if keypoints is not None and keypoints.shape[1] >= len(joint_labels):  
                for i, label in enumerate(joint_labels):
                    x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                    
                    x = None if x == 0 else x
                    y = None if y == 0 else y

                    frame_joint_positions[label + "_x"] = x
                    frame_joint_positions[label + "_y"] = y

                    if debug and frame is not None:
                        if x is not None and y is not None:  
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1) 
                            cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                for label in joint_labels:
                    frame_joint_positions[label + "_x"] = None
                    frame_joint_positions[label + "_y"] = None
        else:
            for label in joint_labels:
                frame_joint_positions[label + "_x"] = None
                frame_joint_positions[label + "_y"] = None

        joint_positions.append(frame_joint_positions)

        # Commented out because it caused a bug on my (Amine) computer, not on Adam's, we never figured out what caused it
        
        # if debug and frame is not None:
        #     cv2.imshow("Joint Positions Debug", frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    if cap:
        cap.release()
        cv2.destroyAllWindows()

    df = pd.DataFrame(joint_positions)
    df.to_csv(output_csv, index=False)



def extract_barbell_positions(video_path, model_barbell, output_csv="barbell_positions.csv", debug=False):
    """
    Extracts barbell bounding box data from a video and saves them to a CSV file. Optionally displays bounding boxes for debugging.

    Parameters:
        video_path (str): Path to the input video.
        model_barbell: The barbell detection model.
        output_csv (str): Path to the output CSV file.
        debug (bool): If True, displays the video with overlaid bounding boxes.

    Outputs:
        Saves a CSV file with barbell bounding box data.
    """
    import cv2
    import pandas as pd

    model_barbell.conf = 0.25  
    model_barbell.iou = 0.5 
    model_barbell.max_det = 10  

    barbell_estim = model_barbell.track(source=video_path, stream=True)
    cap = cv2.VideoCapture(video_path) if debug else None  
    barbell_positions = []

    for frame_id, barbell in enumerate(barbell_estim):
        frame_barbell_positions = {"frame_id": frame_id}

        frame = cap.read()[1] if debug else None

        if hasattr(barbell, "boxes") and len(barbell.boxes) > 0:  
            largest_box = max(barbell.boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
            x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())  
            frame_barbell_positions["barbell_x_min"] = x_min
            frame_barbell_positions["barbell_y_min"] = y_min
            frame_barbell_positions["barbell_x_max"] = x_max
            frame_barbell_positions["barbell_y_max"] = y_max

            if debug and frame is not None:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2) 
                cv2.putText(frame, "Barbell", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            frame_barbell_positions["barbell_x_min"] = None
            frame_barbell_positions["barbell_y_min"] = None
            frame_barbell_positions["barbell_x_max"] = None
            frame_barbell_positions["barbell_y_max"] = None

        barbell_positions.append(frame_barbell_positions)

        # Commented out because it caused a bug on my (Amine) computer, not on Adam's, we never figured out what caused it
        # if debug and frame is not None:
        #     cv2.imshow("Barbell Positions Debug", frame)
        #     if cv2.waitKey(1) & 0xFF == ord('q'):
        #         break

    if cap:
        cap.release()  
        cv2.destroyAllWindows()

    df = pd.DataFrame(barbell_positions)
    df.to_csv(output_csv, index=False)


def process_video(input_path, output_path, output_resolution, output_fps, vertical):
    """
    Process the video: rotate upside down if `vertical` is False, resize to 1280x720, 
    and add black padding to maintain the aspect ratio.
    
    Parameters:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed MPEG4 video.
        vertical (bool): Whether the video is in vertical format.
    """
    target_width, target_height = output_resolution

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {input_path}")
    
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  

    out = cv2.VideoWriter(output_path, fourcc, output_fps, (target_width, target_height))

    scale = min(target_width / input_width, target_height / input_height)
    new_width = int(input_width * scale)
    new_height = int(input_height * scale)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if not vertical:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        padded_frame = cv2.copyMakeBorder(
            resized_frame, pad_y, target_height - new_height - pad_y,
            pad_x, target_width - new_width - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        out.write(padded_frame)
    
    cap.release()
    out.release()

    if not cv2.VideoCapture(output_path).isOpened():
        raise Exception(f"Failed to create a playable video file: {output_path}")

    print(f"Processing completed. Saved to {output_path}")

def is_vertical(yHead, yKnee):
    """
    Determine if the bar starts below the knees at the beginning of the movement.

    Parameters:
    - yHead (list or array): Vertical positions of the bar over time.
    - yKnee (list or array): Vertical positions of the knees over time.

    Returns:
    - bool: True if the head's initial position is above the knees, otherwise False.
    """
    if yHead[0] > yKnee[0]:
        return True
    return False

import numpy as np

def deriv_Barre(yBarre, smoothing=5):
    """
    Calculate the derivative (rate of change) of the bar's vertical position over a specified number of frames.

    Parameters:
    - yBarre (array-like): Vertical positions of the bar over time.
    - smoothing (int): Number of frames over which to compute the derivative, used to reduce noise.

    Returns:
    - np.ndarray: Array of the rate of change (derivative) of yBarre over time.
    """
    yBarre = np.array(yBarre)
    derivatives = np.zeros_like(yBarre, dtype=float)  

    yBarre_filled = np.copy(yBarre)
    for i in range(1, len(yBarre_filled)):
        if np.isnan(yBarre_filled[i]):
            yBarre_filled[i] = yBarre_filled[i - 1]  
    for i in range(len(yBarre) - smoothing):
        derivatives[i] = (yBarre_filled[i + smoothing] - yBarre_filled[i]) / smoothing

    derivatives[-smoothing:] = np.nan

    return derivatives


def bar_direction(yBarre, deriv, i, smoothing, alpha=0.01):
    """
    Determine the direction of the bar's movement (up, down, or still) at a specific point in time.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - deriv (array-like): Derivatives (rate of change) of the bar's vertical positions over time.
    - i (int): The current frame index at which to check the bar's movement.
    - smoothing (int): Number of frames over which to compute the derivative.
    - alpha (float): Threshold ratio for determining significant movement. The derivative must exceed
                     alpha times the vertical range of the derivatives to count as movement.

    Returns:
    - str: "up" if the bar is moving upward.
           "down" if the bar is moving downward.
           "still" if the bar is stationary.
    """
    deriv_range = np.nanmax(np.abs(deriv))  
    threshold = deriv_range * alpha 

    if deriv[i] > threshold:
        return "down"
    elif deriv[i] < -threshold:
        return "up"
    else:
        return "still"


def get_pre_phase(yBarre, deriv, smoothing, alpha=0.01):
    """
    Apply the bar_direction function to all components of the yBarre array.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - deriv (array-like): Derivatives (rate of change) of the bar's vertical positions over time.
    - smoothing (int): Number of frames over which to compute the derivative.
    - alpha (float): Threshold ratio for determining significant movement.

    Returns:
    - np.ndarray: Array of directions ("up", "down", "still") for each frame.
    """
    directions = []
    for i in range(len(yBarre)):
        direction = bar_direction(yBarre, deriv, i, smoothing, alpha)
        directions.append(direction)
    return np.array(directions)


def get_phase(yBarre, yKnee, smoothing, alpha=0.01, N=5):
    """
    Determine the phase of the bar's movement, distinguishing between "still_down" and "still_up".

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - yKnee (list or array): Vertical positions of the knees over time.
    - smoothing (int): Number of frames over which to compute the derivative.
    - alpha (float): Threshold ratio for determining significant movement.
    - N (int): Minimum length of a phase to avoid short interruptions.

    Returns:
    - np.ndarray: Array of phases ("up", "down", "still_down", "still_up") for each frame.
    """
    deriv = deriv_Barre(yBarre, smoothing)

    pre_phases = get_pre_phase(yBarre, deriv, smoothing, alpha)

    phases = []
    for i, phase in enumerate(pre_phases):
        if phase == "still":
            if yBarre[i] > yKnee[i]:
                phases.append("still_down")
            else:
                phases.append("still_up")
        else:
            phases.append(phase)

    phases = np.array(phases) 
    phases = smooth_phases(phases, N)

    return phases


def smooth_phases(phases, N):
    """
    Smooth the phase transitions by removing short interruptions of less than N frames.

    Parameters:
    - phases (np.ndarray): Array of phases ("up", "down", "still_down", "still_up").
    - N (int): Minimum length of a phase to avoid short interruptions.

    Returns:
    - np.ndarray: Smoothed array of phases.
    """
    smoothed_phases = phases.copy()
    current_phase = phases[0]
    start_idx = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            phase_length = i - start_idx
            if phase_length < N:
                smoothed_phases[start_idx:i] = current_phase
            current_phase = phases[i]
            start_idx = i

    phase_length = len(phases) - start_idx
    if phase_length < N:
        smoothed_phases[start_idx:] = current_phase

    return smoothed_phases





def save_phase_video(video_path, phase, sequence_counter, start_frame, end_frame, output_dir, video_properties):
    """
    Save a specific phase segment of the video to a file.
    
    Parameters:
    - video_path (str): Path to the input video file.
    - phase (str): Name of the current phase.
    - sequence_counter (dict): Counter for the phase sequences.
    - start_frame (int): Starting frame of the phase.
    - end_frame (int): Ending frame of the phase.
    - output_dir (str): Directory to save the video segment.
    - video_properties (dict): Properties of the video (fps, width, height, codec).
    """
    sequence_counter[phase] += 1
    output_name = os.path.join(
        output_dir,
        f"{os.path.splitext(os.path.basename(video_path))[0]}_{phase}_{sequence_counter[phase]}.mp4"
    )

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  
    out = cv2.VideoWriter(
        output_name,
        video_properties["fourcc"],
        video_properties["fps"],
        (video_properties["width"], video_properties["height"]),
    )

    for frame_idx in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()
    print(f"Saved: {output_name}")


def separate_phases(video_input, phases, output_dir="output_phases"):
    """
    Separates a video into multiple videos based on phase sequences.

    Parameters:
    - video_input (str): Path to the input video file.
    - phases (list): An array containing the phase name for each frame.
    - output_dir (str): Directory to save the phase-separated videos.

    Outputs:
    - Saves the split videos with names indicating the phase and sequence number.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_input}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if len(phases) != total_frames:
        raise ValueError("The length of the phases array must match the number of frames in the video.")

    video_properties = {"fps": fps, "width": width, "height": height, "fourcc": fourcc}

    sequence_counter = {phase: 0 for phase in set(phases)}
    current_phase = phases[0]
    start_frame = 0

    for i in range(1, len(phases)):
        if phases[i] != current_phase:
            save_phase_video(
                video_input, current_phase, sequence_counter,
                start_frame, i - 1, output_dir, video_properties
            )
            current_phase = phases[i]
            start_frame = i

    save_phase_video(
        video_input, current_phase, sequence_counter,
        start_frame, len(phases) - 1, output_dir, video_properties
    )

    cap.release()
    print(f"All phases have been separated and saved in {output_dir}.")




def estimate_movement(video_path, pose_detection_model, barbell_detection_model, debug=False):
    """
    Estimates the movement phase of a deadlift by processing the video.
    It uses YOLO models to detect pose and barbell at each frame, constructs sequences of frames
    and corresponding keypoints+barbell coordinates, and then runs inference using the trained classifier.

    Parameters:
        video_path (str): Path to the input video file.
        pose_detection_model: The YOLO pose detection model used during training.
        barbell_detection_model: The YOLO barbell detection model used during training.
        debug (bool): If True, prints detailed information per frame.

    Returns:
        str: The predicted movement phase label.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if "still_up" in video_path:
        model_path = "models/best_deadlift_lockout_phase_classifier.pth"
        phase_key = "lockout"
        class_labels = [
            "Incomplete lockout due to insufficient glute engagement.",
            "Shoulders are drifting too far behind the bar. Align them vertically.",
            "Lockout is correct."
        ]
    elif "still_down" in video_path:
        model_path = "models/best_deadlift_setup_phase_classifier.pth"
        phase_key = "setup"
        class_labels = [
            "Setup error: Hips are too low (like a squat).",
            "Setup error: Rounded upper back (thoracic spine). Maintain a neutral spine.",
            "Setup phase is correct."
        ]
    elif "up" in video_path and "still_up" not in video_path:
        model_path = "models/best_deadlift_ascending_phase_classifier.pth"
        phase_key = "ascending"
        class_labels = [
            "Ascending error: Rounded lower back (lumbar spine). Maintain a neutral spine to avoid injury.",
            "Ascending error: The barbell is drifting away from the body. Keep it close to your shins.",
            "Ascending phase is correct."
        ]
    elif "down" in video_path and "still_down" not in video_path:
        model_path = "models/best_deadlift_descending_phase_classifier.pth"
        phase_key = "descending"
        class_labels = [
            "Descending error: Barbell is drifting away from the legs. Keep it close to your shins.",
            "Descending error: Rounded lower back. Maintain a neutral spine during the descent.",
            "Descending phase is correct."
        ]
    else:
        raise ValueError("Video path does not contain recognizable phase keywords.")

    num_keypoints = 36 
    pretrained_visual_model_name = "google/vit-base-patch16-224-in21k"
    latent_dim = 256
    num_visual_tokens = 16
    num_classes = len(class_labels)
    seq_length = 5  

    inference_model = DeadliftMovementClassifier(
        num_keypoints,
        pretrained_visual_model_name,
        latent_dim,
        num_visual_tokens,
        num_classes
    ).to(device)
    inference_model.load_state_dict(torch.load(model_path, map_location=device))
    inference_model.eval()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    frame_buffer = []
    all_logits = []
    frame_id = 0

    with torch.no_grad():
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            pose_output = pose_detection_model(frame, verbose = debug)
            barbell_output = barbell_detection_model(frame, verbose = debug)

            if len(pose_output) > 0:
                pose = pose_output[0]
            else:
                pose = None

            sorted_people = sorted(
                pose.keypoints if pose and hasattr(pose, "keypoints") else [],
                key=lambda p: p.box.area if hasattr(p, "box") else 0,
                reverse=True
            )
            if not sorted_people:
                if debug:
                    print(f"Frame {frame_id}: No person detected.")
                frame_id += 1
                continue

            person = sorted_people[0]
            keypoints = person.xy if hasattr(person, "xy") else None
            if keypoints is None or len(keypoints) == 0:
                if debug:
                    print(f"Frame {frame_id}: No valid keypoints detected.")
                frame_id += 1
                continue

            if len(barbell_output) > 0 and hasattr(barbell_output[0], "boxes") and len(barbell_output[0].boxes) > 0:
                barbell = barbell_output[0]
                largest_box = max(
                    barbell.boxes,
                    key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1])
                )
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                barbell_coords = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            else:
                if debug:
                    print(f"Frame {frame_id}: No barbell detected.")
                frame_id += 1
                continue  

            keypoints = torch.tensor(keypoints, device=device)
            barbell_tensor = torch.tensor(barbell_coords, device=device).unsqueeze(0).unsqueeze(0)  

            combined_vector = torch.cat((keypoints, barbell_tensor), dim=1) 

            frame_buffer.append((frame, combined_vector))

            if debug:
                print(f"Frame {frame_id}: Detected person and barbell.")

            if len(frame_buffer) == seq_length:
                images_seq, kpts_seq = zip(*frame_buffer)
                
                images_seq = torch.stack([
                    torch.tensor(img, dtype=torch.float32, device=device).permute(2, 0, 1) 
                    for img in images_seq
                ])
                images_seq = images_seq.unsqueeze(0) 

                kpts_seq = torch.stack(kpts_seq, dim=0).to(device)  
                kpts_seq = kpts_seq.view(1, seq_length, -1) 

                logits = inference_model(kpts_seq, images_seq)
                all_logits.append(logits.cpu())

                if debug:
                    print(f"Processed sequence ending at frame {frame_id}.")

                frame_buffer.pop(0)

            frame_id += 1

    cap.release()

    if not all_logits:
        return "No valid predictions."

    avg_logits = torch.mean(torch.cat(all_logits, dim=0), dim=0, keepdim=True)
    predicted_class_idx = torch.argmax(avg_logits, dim=1).item()
    movement_result = class_labels[predicted_class_idx]

    if debug:
        print(f"Final Movement Result: {movement_result}")

    return movement_result

## auxiliary functions, namely those used to label the videos at 90 degrees points of view.

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
