import pandas as pd
import cv2
import numpy as np
import os

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
    cap = cv2.VideoCapture(video_path) if debug else None  # Open video for debugging if needed

    joint_positions = []

    for frame_id, pose in enumerate(pose_estim):
        frame_joint_positions = {"frame_id": frame_id}

        # Get the frame for debugging
        frame = cap.read()[1] if debug else None

        # Process pose estimation (joints)
        sorted_people = sorted(pose.keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True)
        if sorted_people:
            person = sorted_people[0]
            keypoints = person.xy if hasattr(person, "xy") else None

            if keypoints is not None and keypoints.shape[1] >= len(joint_labels):  # Check if keypoints are valid
                for i, label in enumerate(joint_labels):
                    x, y = int(keypoints[0][i][0]), int(keypoints[0][i][1])
                    
                    # Replace 0 values with None
                    x = None if x == 0 else x
                    y = None if y == 0 else y

                    frame_joint_positions[label + "_x"] = x
                    frame_joint_positions[label + "_y"] = y

                    if debug and frame is not None:
                        if x is not None and y is not None:  # Only draw if coordinates are valid
                            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)  # Green circle
                            cv2.putText(frame, label, (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            else:
                # If no keypoints are detected, set joint positions to None
                for label in joint_labels:
                    frame_joint_positions[label + "_x"] = None
                    frame_joint_positions[label + "_y"] = None
        else:
            # If no person is detected, set all joint positions to None
            for label in joint_labels:
                frame_joint_positions[label + "_x"] = None
                frame_joint_positions[label + "_y"] = None

        joint_positions.append(frame_joint_positions)

        # Show debug video if enabled
        if debug and frame is not None:
            cv2.imshow("Joint Positions Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap:
        cap.release()  # Release video capture if debug mode is enabled
        cv2.destroyAllWindows()

    # Save to CSV
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

    # Optional: Modify the model inference time
    model_barbell.conf = 0.25  # Set confidence threshold to a lower value for better estimation
    model_barbell.iou = 0.5   # Set IoU threshold to balance precision and recall
    model_barbell.max_det = 10  # Allow more detections for evaluation

    barbell_estim = model_barbell.track(source=video_path, stream=True)
    cap = cv2.VideoCapture(video_path) if debug else None  # Open video for debugging if needed

    barbell_positions = []

    for frame_id, barbell in enumerate(barbell_estim):
        frame_barbell_positions = {"frame_id": frame_id}

        # Get the frame for debugging
        frame = cap.read()[1] if debug else None

        # Process barbell detection
        if hasattr(barbell, "boxes") and len(barbell.boxes) > 0:  # Check if boxes exist
            # Find the largest barbell box by area
            largest_box = max(barbell.boxes, key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
            x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())  # Convert tensor to list if necessary
            frame_barbell_positions["barbell_x_min"] = x_min
            frame_barbell_positions["barbell_y_min"] = y_min
            frame_barbell_positions["barbell_x_max"] = x_max
            frame_barbell_positions["barbell_y_max"] = y_max

            if debug and frame is not None:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)  # Red rectangle
                cv2.putText(frame, "Barbell", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        else:
            # If no barbell is detected, set bounding box coordinates to None
            frame_barbell_positions["barbell_x_min"] = None
            frame_barbell_positions["barbell_y_min"] = None
            frame_barbell_positions["barbell_x_max"] = None
            frame_barbell_positions["barbell_y_max"] = None

        barbell_positions.append(frame_barbell_positions)

        # Show debug video if enabled
        if debug and frame is not None:
            cv2.imshow("Barbell Positions Debug", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    if cap:
        cap.release()  # Release video capture if debug mode is enabled
        cv2.destroyAllWindows()

    # Save to CSV
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
    # Desired output resolution
    target_width, target_height = output_resolution

    # Open the video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {input_path}")
    
    # Get input video properties
    input_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    input_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MPEG4 codec

    # Set up video writer
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (target_width, target_height))

    # Determine scaling and padding
    scale = min(target_width / input_width, target_height / input_height)
    new_width = int(input_width * scale)
    new_height = int(input_height * scale)
    pad_x = (target_width - new_width) // 2
    pad_y = (target_height - new_height) // 2

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Rotate frame upside down if `vertical` is False
        if not vertical:
            frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Resize frame while preserving aspect ratio
        resized_frame = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Add black padding to make it 1280x720
        padded_frame = cv2.copyMakeBorder(
            resized_frame, pad_y, target_height - new_height - pad_y,
            pad_x, target_width - new_width - pad_x,
            cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )

        # Write the processed frame to the output video
        out.write(padded_frame)
    
    # Release resources
    cap.release()
    out.release()

    # Ensure output video file is valid
    if not cv2.VideoCapture(output_path).isOpened():
        raise Exception(f"Failed to create a playable video file: {output_path}")

    print(f"Processing completed. Saved to {output_path}")



def deriv_Barre(yBarre, smoothing=5):
    """
    Calculate the derivative (rate of change) of the bar's vertical position over a specified number of frames.

    Parameters:
    - yBarre (array-like): Vertical positions of the bar over time.
    - smoothing (int): Number of frames over which to compute the derivative, used to reduce noise.

    Returns:
    - np.ndarray: Array of the rate of change (derivative) of yBarre over time.
    """
    yBarre = np.array(yBarre)  # Ensure input is a NumPy array
    derivatives = np.zeros_like(yBarre)  # Initialize array for derivatives
    
    # Compute the derivative over the valid range
    for i in range(len(yBarre) - smoothing):
        derivatives[i] = (yBarre[i + smoothing] - yBarre[i]) / smoothing
    
    # Fill the remaining positions with NaN or 0 if smoothing window exceeds bounds
    derivatives[-smoothing:] = np.nan  # Optional: change np.nan to 0 if needed

    return derivatives

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

def bar_direction(yBarre, i, smoothing, y1, y2, alpha=0.01):
    """
    Determine the direction of the bar's movement (up, down, or still) at a specific point in time.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - i (int): The current frame index at which to check the bar's movement.
    - smoothing (int): Number of frames over which to compute the derivative.
    - y1 (list or array): Vertical positions of the first body part (e.g., shoulders).
    - y2 (list or array): Vertical positions of the second body part (e.g., waist).
    - alpha (float): Threshold ratio for determining significant movement. The derivative must exceed
                     alpha times the vertical distance between y1 and y2 to count as movement.

    Returns:
    - str: "up" if the bar is moving upward.
           "down" if the bar is moving downward.
           "still" if the bar is stationary.
    """

    treshold = abs(y2[i] - y1[i]) * alpha  # Threshold based on the vertical distance between y1 and y2
    if deriv_Barre(yBarre, i, smoothing) > treshold:
        return "up"
    elif deriv_Barre(yBarre, i, smoothing) < -treshold:
        return "down"
    else:
        return "still"
    
def get_pre_phase(yBarre, smoothing, y1, y2, alpha=0.01):
    """
    Apply the bar_direction function to all components of the yBarre array.

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - smoothing (int): Number of frames over which to compute the derivative.
    - y1 (list or array): Vertical positions of the first body part (e.g., shoulders).
    - y2 (list or array): Vertical positions of the second body part (e.g., waist).
    - alpha (float): Threshold ratio for determining significant movement.

    Returns:
    - np.ndarray: Array of directions ("up", "down", "still") for each frame.
    """

    directions = []
    for i in range(len(yBarre)):
        direction = bar_direction(yBarre, i, smoothing, y1, y2, alpha)
        directions.append(direction)
    return np.array(directions)

def get_phase(yBarre, yKnee, smoothing, y1, y2, alpha=0.01):
    """
    Determine the phase of the bar's movement, distinguishing between "still_down" and "still_up".

    Parameters:
    - yBarre (list or array): Vertical positions of the bar over time.
    - yKnee (list or array): Vertical positions of the knees over time.
    - smoothing (int): Number of frames over which to compute the derivative.
    - y1 (list or array): Vertical positions of the first body part (e.g., shoulders).
    - y2 (list or array): Vertical positions of the second body part (e.g., waist).
    - alpha (float): Threshold ratio for determining significant movement.

    Returns:
    - np.ndarray: Array of phases ("up", "down", "still_down", "still_up") for each frame.
    """
    pre_phases = get_pre_phase(yBarre, smoothing, y1, y2, alpha)
    phases = []

    for i, phase in enumerate(pre_phases):
        if phase == "still":
            if yBarre[i] > yKnee[i]:
                phases.append("still_down")
            else:
                phases.append("still_up")
        else:
            phases.append(phase)

    return np.array(phases)


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

    # Set up video writer
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Go to the starting frame
    out = cv2.VideoWriter(
        output_name,
        video_properties["fourcc"],
        video_properties["fps"],
        (video_properties["width"], video_properties["height"]),
    )

    # Write frames from start_frame to end_frame
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
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video and get its properties
    cap = cv2.VideoCapture(video_input)
    if not cap.isOpened():
        raise Exception(f"Cannot open video file: {video_input}")

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check if the phases array matches the number of frames
    if len(phases) != total_frames:
        raise ValueError("The length of the phases array must match the number of frames in the video.")

    # Store video properties for convenience
    video_properties = {"fps": fps, "width": width, "height": height, "fourcc": fourcc}

    # Initialize phase tracking
    sequence_counter = {phase: 0 for phase in set(phases)}
    current_phase = phases[0]
    start_frame = 0

    # Iterate through the phases array
    for i in range(1, len(phases)):
        if phases[i] != current_phase:  # Phase change detected
            save_phase_video(
                video_input, current_phase, sequence_counter,
                start_frame, i - 1, output_dir, video_properties
            )
            current_phase = phases[i]
            start_frame = i

    # Save the last segment
    save_phase_video(
        video_input, current_phase, sequence_counter,
        start_frame, len(phases) - 1, output_dir, video_properties
    )

    cap.release()
    print(f"All phases have been separated and saved in {output_dir}.")
