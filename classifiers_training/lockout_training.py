import os
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from ultralytics import YOLO  # Import YOLO
from main_model import DeadliftMovementClassifier, DeadliftDataset
import cv2  # For video processing
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, f1_score, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import pandas as pd
import numpy as np

# Define YOLO models for pose and barbell detection
pose_detection_model = YOLO("../models/yolo11x-pose.pt")
barbell_detection_model = YOLO("../models/best_barbell_detector_bar.pt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Folder paths for labeled data (relative paths)
folder_paths_train = {
    "Lockout error: Incomplete lockout due to insufficient glute engagement.": "../processed_videos/train/Lockout error: Incomplete lockout due to insufficient glute engagement.",
    "Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.": "../processed_videos/train/Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.",
    "Lockout is correct": "../processed_videos/train/Lockout is correct"
}

folder_paths_test = {
    "Lockout error: Incomplete lockout due to insufficient glute engagement.": "../processed_videos/test/Lockout error: Incomplete lockout due to insufficient glute engagement.",
    "Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.": "../processed_videos/test/Lockout error: Shoulders are drifting too far behind the bar. Align them vertically.",
    "Lockout is correct": "../processed_videos/test/Lockout is correct"
}

# Hyperparameters
num_epochs = 20
batch_size = 8
learning_rate = 1e-4

# Instantiate datasets
print("Creating training dataset...")
start_time = time.time()
seq_length = 5
train_dataset = DeadliftDataset(folder_paths_train, seq_length=seq_length, pose_detection_model = pose_detection_model, barbell_detection_model = barbell_detection_model)
print(f"Training dataset created in {time.time() - start_time:.2f} seconds.")

print("Creating test dataset...")
start_time = time.time()
test_dataset = DeadliftDataset(folder_paths_test, seq_length=seq_length, pose_detection_model = pose_detection_model, barbell_detection_model = barbell_detection_model)
print(f"Test dataset created in {time.time() - start_time:.2f} seconds.")

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Model Configuration
num_keypoints = 36  # e.g., 18 joints with x, y coordinates + 2 barbell coordinates
pretrained_visual_model_name = "google/vit-base-patch16-224-in21k"  # Vision Transformer
latent_dim = 256
num_visual_tokens = 16
num_classes = len(folder_paths_train)  # Number of classes (3 in this case)

# Instantiate Model
model = DeadliftMovementClassifier(num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes)
model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 1e-5)

best_accuracy = 0.0  
best_model_path = "best_deadlift_lockout_phase_classifier.pth"
best_loss = 1000000

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for keypoints, images, labels in train_loader:
        images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)
        optimizer.zero_grad()

        logits = model(keypoints, images)
        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    average_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {average_loss:.4f}")

    # Evaluate on test set every 2 epochs
    if (epoch + 1) % 2 == 0:
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels_list = []
        all_probs = []
        with torch.no_grad():
            for keypoints, images, labels in test_loader:
                images, keypoints, labels = images.to(device), keypoints.to(device), labels.to(device)
                logits = model(keypoints, images)
                probabilities = torch.softmax(logits, dim=1)
                predictions = torch.argmax(probabilities, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                all_preds.extend(predictions.cpu().numpy())
                all_labels_list.extend(labels.cpu().numpy())
                all_probs.extend(probabilities.cpu().numpy())

        accuracy = correct / total
        print(f"Epoch {epoch + 1}/{num_epochs}, Test Accuracy: {accuracy * 100:.2f}%")

        if accuracy > best_accuracy or (accuracy == best_accuracy and average_loss < best_loss):
            best_accuracy = accuracy
            best_loss = average_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

print(f"Training completed. Best Test Accuracy: {best_accuracy * 100:.2f}%")
