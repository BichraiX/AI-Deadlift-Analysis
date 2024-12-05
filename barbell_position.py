### This program is supposed to create a csv with the positions of the barbell in the video.

from ultralytics import YOLO
import cv2

# Load the model
model = YOLO('best.pt')  # Replace 'best.pt' with your fine-tuned YOLO model's path

# Path to the video
video_path = 'Worst deadlift form ever [QfP02tIWoPU].mp4'
output_video_path = 'output_with_detections.mp4'

# Open the video
cap = cv2.VideoCapture(video_path)
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Video writer to save output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the current frame
    results = model(frame)

    # Extract detections
    for result in results:  # Iterate through each result
        for box in result.boxes:  # Loop over all detected boxes
            # Extract bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0].tolist()  # Coordinates as floats
            conf = box.conf[0] if box.conf is not None else None  # Confidence score
            cls = box.cls[0] if box.cls is not None else None  # Class index

            if conf and conf > 0.5:  # Apply confidence threshold
                # Draw the bounding box
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"Barbell {conf:.2f}" if conf else "Barbell"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Save the processed frame to output video
    out.write(frame)

    # Optional: Show the frame (useful for debugging)
    # cv2.imshow('YOLO Detection', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
