from ultralytics import YOLO
import cv2

# Load your YOLO model
model = YOLO("runs/detect/train/weights/best.pt")  # Replace with the path to your trained model
# Path to your video file
video_path = "Worst deadlift form ever [QfP02tIWoPU].mp4"

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video was loaded
if not cap.isOpened():
    print("Error: Unable to open video file.")
while cap.isOpened():
    ret, frame = cap.read()  # Read a frame
    if not ret:
        break  # Exit loop if no frame is returned (end of video)
    
    # Run YOLO inference on the frame
    results = model.predict(source=frame, imgsz=640, conf=0.25, verbose=False)

    # Extract detections for the barbell
    for result in results:
        boxes = result.boxes  # Get bounding boxes
        
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]  # Bounding box coordinates
            y_center = (y1 + y2) / 2  # Calculate the vertical position (y-center)
            print(f"Barbell y-coordinate: {y_center}")

            # Optional: Draw the bounding box and display it
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.circle(frame, (int((x1 + x2) / 2), int(y_center)), 5, (0, 0, 255), -1)

    # Display the frame (optional)
    cv2.imshow("Barbell Detection", frame)

    # Break on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Define the output video writer
output_path = "output_video.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), 
                      (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

# Inside the frame processing loop, write each frame to the output video
out.write(frame)

# Release the video writer
out.release()
