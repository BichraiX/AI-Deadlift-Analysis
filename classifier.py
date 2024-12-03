import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.io import read_video

import open_clip

import os

class VideoCLIP(nn.Module):
    def __init__(self, clip_model, num_frames=16):
        super(VideoCLIP, self).__init__()
        self.clip_model = clip_model
        self.num_frames = num_frames
        self.frame_agg = nn.Linear(clip_model.visual.output_dim, clip_model.visual.output_dim) # Linear layer for aggregation

    def forward(self, video_frames):
        # video_frames: shape [num_frames, 3, 224, 224]
        frame_embeddings = []

        # Encode each frame and aggregate
        for frame in video_frames:
            frame = frame.unsqueeze(0)  # Add batch dimension
            frame_embedding = self.clip_model.encode_image(frame)  # [1, embed_dim]
            frame_embeddings.append(frame_embedding)
        
        # Aggregate frame embeddings
        frame_embeddings = torch.cat(frame_embeddings, dim=0)  # Shape: [num_frames, embed_dim]
        video_embedding = self.frame_agg(frame_embeddings.mean(dim=0, keepdim=True))  # Shape: [1, embed_dim]

        return video_embedding

class DeadliftVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, frames_per_video=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.label_to_idx = {
            'good movement' : 0,
            'bad movement' : 1
        }
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.label_to_idx[self.labels[idx]]
        
        # Read video frames
        video_frames, _, info = read_video(video_path, pts_unit='sec')
        total_frames = video_frames.shape[0]

        # Sample frames evenly throughout the video
        indices = torch.linspace(0, total_frames - 1, self.frames_per_video).long()
        sampled_frames = video_frames[indices]
        
        # Apply transforms to frames
        if self.transform:
            frames = [self.transform(frame) for frame in sampled_frames]
        else:
            frames = [frame for frame in sampled_frames]

        # Stack frames into a tensor
        frames_tensor = torch.stack(frames)  # Shape: [frames_per_video, C, H, W]

        return frames_tensor, label


class DeadliftVideoDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, augmentation=None, frames_per_video=16):
        self.video_paths = video_paths
        self.labels = labels
        self.transform = transform
        self.augmentation = augmentation
        self.frames_per_video = frames_per_video
        self.label_to_idx = {
            'bad movement': 0,
            'good movement': 1
        }
        
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.label_to_idx[self.labels[idx]]
        
        # Read video frames
        video_frames, _, info = read_video(video_path, pts_unit='sec')
        total_frames = video_frames.shape[0]

        # Sample frames evenly throughout the video
        indices = torch.linspace(0, total_frames - 1, self.frames_per_video).long()
        sampled_frames = video_frames[indices]
        
        # Apply transforms to frames
        frames = []
        for frame in sampled_frames:
            frame = frame.permute(2, 0, 1)  # Convert to (C, H, W)
            if self.transform:
                frame = self.transform(frame)
            if self.augmentation:
                frames.append(self.augmentation(frame))
            frames.append(frame)
    
        # Stack frames into a tensor
        frames_tensor = torch.stack(frames)  # Shape: [frames_per_video, C, H, W]
    
        return frames_tensor, label

model_name = 'ViT-B-32'  # You can choose other architectures
pretrained = 'openai'

model, _, _ = open_clip.create_model_and_transforms(
    model_name=model_name,
    pretrained=pretrained
)
model.eval()  # Set to evaluation mode

class VideoCLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(VideoCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.fc = nn.Linear(clip_model.visual.output_dim, num_classes)
        
    def forward(self, frames):
        # frames: [batch_size, frames_per_video, C, H, W]
        batch_size, frames_per_video, C, H, W = frames.shape
        frames = frames.view(-1, C, H, W)  # Flatten frames
        with torch.no_grad():
            frame_features = self.clip_model.encode_image(frames)  # [batch_size * frames_per_video, output_dim]
        frame_features = frame_features.view(batch_size, frames_per_video, -1)
        video_features = frame_features.mean(dim=1)  # Average over frames
        logits = self.fc(video_features)  # [batch_size, num_classes]
        return logits

class VideoCLIPClassifierWithAttention(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(VideoCLIPClassifierWithAttention, self).__init__()
        self.clip_model = clip_model
        self.num_classes = num_classes
        self.attention_layer = nn.MultiheadAttention(embed_dim=clip_model.visual.output_dim, num_heads=4)
        self.fc = nn.Linear(clip_model.visual.output_dim, num_classes)
    
    def forward(self, frames):
        batch_size, frames_per_video, C, H, W = frames.shape
        frames = frames.view(-1, C, H, W)  # Flatten frames
        with torch.no_grad():
            frame_features = self.clip_model.encode_image(frames)
        frame_features = frame_features.view(batch_size, frames_per_video, -1)  # [batch_size, frames_per_video, output_dim]

        # Attention over frames
        frame_features = frame_features.permute(1, 0, 2)  # [frames_per_video, batch_size, output_dim]
        attn_output, _ = self.attention_layer(frame_features, frame_features, frame_features)
        video_features = attn_output.mean(dim=0)  # [batch_size, output_dim]
        logits = self.fc(video_features)
        return logits



num_classes = 2  # Number of movement labels
classifier = VideoCLIPClassifierWithAttention(model, num_classes)

import os
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

dataset_root = './dataset/Adam'

video_paths = []
labels = []

for label_folder in ['bad', 'good']:
    label_path = os.path.join(dataset_root, label_folder)
    
    if not os.path.isdir(label_path):
        print(f"Folder '{label_folder}' does not exist in '{dataset_root}'")
        continue

    label = 'bad movement' if label_folder == 'bad' else 'good movement'
    
    for video_file in os.listdir(label_path):
        if video_file.endswith('.mp4'):  
            video_paths.append(os.path.join(label_path, video_file))
            labels.append(label)

print("Video Paths:", video_paths)
print("Labels:", labels)

train_video_paths, test_video_paths, train_labels, test_labels = train_test_split(
    video_paths, labels, test_size=0.4, stratify=labels, random_state=42
)

train_augmentation = transforms.RandomApply([
    transforms.RandomHorizontalFlip(p=0.5),       # Flip horizontally
    transforms.RandomRotation(degrees=15),        # Rotate randomly within ±15 degrees
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Adjust brightness and contrast
    transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),  # Random crop and resize
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # Small translations and rotations
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),  # Apply Gaussian blur
    transforms.RandomPerspective(distortion_scale=0.5, p=0.5),   # Perspective transformation
    transforms.RandomGrayscale(p=0.2),   # Convert some images to grayscale
    transforms.RandomErasing(scale=(0.02, 0.2), ratio=(0.3, 3.3), p=0.2),  # Randomly erase parts of the image
], p=0.7)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ConvertImageDtype(torch.float32),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711)
    ),
])

train_dataset = DeadliftVideoDataset(train_video_paths, train_labels, transform=transform, augmentation=train_augmentation)
test_dataset = DeadliftVideoDataset(test_video_paths, test_labels, transform=transform)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
classifier = classifier.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(classifier.fc.parameters(), lr=1e-4)  # Only train the classification head


num_epochs = 1000

for epoch in range(num_epochs):
    classifier.train()
    running_loss = 0.0
    for i, (frames, labels) in enumerate(train_dataloader):
        frames = frames.to(device)  # [batch_size, frames_per_video, C, H, W]
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = classifier(frames)  # [batch_size, num_classes]
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if i % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
    if epoch%20 == 0 :
        torch.save(classifier.state_dict(), 'deadlift_classifier.pth')
    epoch_loss = running_loss / len(train_dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')


classifier.load_state_dict(torch.load('deadlift_classifier.pth'))
classifier.eval()


correct = 0
total = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for frames, labels in test_dataloader:
        frames = frames.to(device)
        labels = labels.to(device)
        outputs = classifier(frames)
        _, predicted = torch.max(outputs, 1)

        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total * 100
print(f'Accuracy on the test set: {accuracy:.2f}%')

idx_to_label = {v: k for k, v in test_dataset.label_to_idx.items()}
all_predictions_labels = [idx_to_label[pred] for pred in all_predictions]
all_labels_labels = [idx_to_label[label] for label in all_labels]

for pred, true_label in zip(all_predictions_labels, all_labels_labels):
    print(f'Predicted: {pred}, Actual: {true_label}')
