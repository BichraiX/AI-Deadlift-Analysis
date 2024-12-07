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

# Folder paths for labeled data
folder_paths = {
    "Incomplete lockout due to insufficient glute engagement.": "/Users/aminechraibi/Desktop/Projet CV/Test Yolo/processed_videos/Lockout error: Incomplete lockout due to insufficient glute engagement."
    "Shoulders are drifting too far behind the bar. Align them vertically.": "/Users/aminechraibi/Desktop/Projet CV/Test Yolo/processed_videos/Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.", 
    "Lockout is correct": "/Users/aminechraibi/Desktop/Projet CV/Test Yolo/processed_videos/Lockout is correct", 
}

# Dataset class
class DeadliftDataset(Dataset):
    def __init__(self, folder_paths, transform=None):
        self.data = []
        self.labels = []
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

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Pose detection
            pose_result = pose_detection_model(frame)

            # Barbell detection
            barbell_result = barbell_detection_model(frame)

            # Extract keypoints
            if len(pose_result.xy) > 0:
                keypoints = pose_result.xy[0].tolist()  # Assuming one person per frame
            else:
                continue  # Skip if no keypoints detected

            # Extract barbell coordinates
            if barbell_result.boxes:
                largest_box = max(
                    barbell_result.boxes,
                    key=lambda box: (box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]),
                )
                x_min, y_min, x_max, y_max = map(int, largest_box.xyxy[0].tolist())
                barbell_coords = [(x_min + x_max) / 2, (y_min + y_max) / 2]
            else:
                continue  # Skip if no barbell detected

            # Combine keypoints and barbell coordinates
            combined_vector = keypoints + barbell_coords

            # Append to dataset
            self.data.append((frame, torch.tensor(combined_vector, dtype=torch.float32)))
            self.labels.append(label_idx)

        cap.release()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, keypoints = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return keypoints, image, label


# Hyperparameters
num_epochs = 10
batch_size = 8
learning_rate = 1e-4

# Instantiate dataset
print("Creating dataset...")
start_time = time.time()  # Record the start time
dataset = DeadliftDataset(folder_paths)
end_time = time.time()  # Record the end time
print(f"Dataset created in {end_time - start_time:.2f} seconds.")

# Split dataset into train and test sets
test_size = 0.2
train_dataset, test_dataset = train_test_split(dataset, test_size=test_size)

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Configuration
num_keypoints = 19  # e.g., 18 joints with x, y coordinates + 2 barbell coordinates
pretrained_visual_model_name = "google/vit-base-patch16-224-in21k"  # Vision Transformer
latent_dim = 256
num_visual_tokens = 16
num_classes = len(folder_paths)  # Number of classes (3 in this case)

# Instantiate Model
model = DeadliftMovementClassifier(num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for keypoints, images, labels in train_loader:
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

# Evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for keypoints, images, labels in test_loader:
        logits = model(keypoints, images)
        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
