import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO  # Import YOLO
from perceiver_based_classifier import DeadliftMovementClassifier
import cv2  # For video processing
import time

# Define YOLO models for pose and barbell detection
pose_detection_model = YOLO("../models/yolo11x-pose.pt")
barbell_detection_model = YOLO("../models/best_barbell_detector_bar.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Folder paths for labeled data
folder_paths = {
    "Incomplete lockout due to insufficient glute engagement.": "/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/processed_videos/Lockout error: Incomplete lockout due to insufficient glute engagement.",
    "Shoulders are drifting too far behind the bar. Align them vertically.": "/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/processed_videos/Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.", 
    "Lockout is correct": "/users/eleves-a/2022/amine.chraibi/AI-Deadlift-Analysis/processed_videos/Lockout is correct", 
}

class DeadliftDataset(Dataset):
    def __init__(self, folder_paths, seq_length, transform=None):
        self.data = []
        self.labels = []
        self.seq_length = seq_length  # Number of frames per sequence
        self.transform = transform

        # Load data and labels
        for label_idx, (label_name, folder_path) in enumerate(folder_paths.items()):
            for file_name in os.listdir(folder_path):
                file_path = os.path.join(folder_path, file_name)
                self.process_video(file_path, label_idx)

    def process_video(self, file_path, label_idx):
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return

        frame_buffer = []  # Buffer to hold frames and keypoints for sequence creation
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pose detection
            pose_result = pose_detection_model(frame)

            # Barbell detection
            barbell_result = barbell_detection_model(frame)

            # Extract keypoints
            sorted_people = sorted(
                pose_result[0].keypoints, key=lambda p: p.box.area if hasattr(p, "box") else 0, reverse=True
            )
            if not sorted_people:
                continue
            person = sorted_people[0]
            keypoints = person.xy if hasattr(person, "xy") else None
            if keypoints is None or len(keypoints) == 0:
                continue

            # Extract barbell coordinates
            if barbell_result[0].boxes:
                largest_box = max(
                    barbell_result[0].boxes,
                    key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]),
                )
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                barbell_coords = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            else:
                continue  # Skip if no barbell detected
            keypoints = keypoints.to(device)
            # Combine keypoints and barbell coordinates
            combined_vector = torch.cat((keypoints, torch.tensor(barbell_coords, device = device).unsqueeze(0).unsqueeze(0)), dim = 1)
            # Add to buffer
            frame_buffer.append((frame, combined_vector))

            # If buffer has enough frames for a sequence, save the sequence
            if len(frame_buffer) == self.seq_length:
                images, keypoints_sequence = zip(*frame_buffer)  # Unpack buffer
                images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])  # Stack images
                keypoints_sequence = torch.stack(keypoints_sequence, dim=0)   
                self.data.append((images, keypoints_sequence))
                self.labels.append(label_idx)
                frame_buffer.pop(0)  # Slide window forward

        cap.release()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, keypoints = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            images = self.transform(images)

        return keypoints, images, label


# Hyperparameters
num_epochs = 20
batch_size = 8
learning_rate = 1e-4

# Instantiate dataset
print("Creating dataset...")
start_time = time.time()  # Record the start time
seq_length = 5 
dataset = DeadliftDataset(folder_paths, seq_length = seq_length)
end_time = time.time()  # Record the end time
print(f"Dataset created in {end_time - start_time:.2f} seconds.")

# Split dataset into train and test sets
test_size = 0.2
train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Configuration
num_keypoints = 36  # e.g., 18 joints with x, y coordinates + 2 barbell coordinates
pretrained_visual_model_name = "google/vit-base-patch16-224-in21k"  # Vision Transformer
latent_dim = 256
num_visual_tokens = 16
num_classes = len(folder_paths)  # Number of classes (3 in this case)

# Instantiate Model
model = DeadliftMovementClassifier(num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes)
model.to(device)
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_accuracy = 0.0  # Variable to store the best accuracy
best_model_path = "best_deadlift_lockout_phase_classifier.pth"
best_loss = 1000000
# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for keypoints, images, labels in train_loader:
        images = images.to(device)
        keypoints = keypoints.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(keypoints, images)

        # Compute loss
        loss = criterion(logits, labels)
        total_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader)}")

    # Test the model every 2 epochs
    if (epoch + 1) % 2 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for keypoints, images, labels in test_loader:
                images = images.to(device)
                keypoints = keypoints.to(device)
                labels = labels.to(device)
                
                # Forward pass
                logits = model(keypoints, images)
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy * 100:.2f}%")

        # Save the model if it has the best accuracy
        if accuracy >= best_accuracy and total_loss / len(train_loader) < best_loss / len(train_loader):
            best_accuracy = accuracy
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

print(f"Training completed. Best Test Accuracy: {best_accuracy * 100:.2f}%")