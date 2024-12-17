import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig, AutoModel
import torch.nn.functional as F
from torch.utils.data import Dataset
import os
import cv2

class DeadliftMovementClassifier(nn.Module):
    def __init__(self, num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes):
        super(DeadliftMovementClassifier, self).__init__()

        ### 1. Pretrained Visual Encoder ###
        self.visual_encoder = AutoModel.from_pretrained(pretrained_visual_model_name)
        for param in self.visual_encoder.parameters():
            param.requires_grad = False
        visual_output_dim = self.visual_encoder.config.hidden_size

        ### 2. Perceiver Resampler ###
        self.perceiver_resampler = PerceiverModel(
            PerceiverConfig(
                input_dim=visual_output_dim,
                num_latents=num_visual_tokens,
                d_latents=latent_dim,
                num_self_attention_layers=2
            )
        )

        ### 3. Keypoint Encoder ###
        self.keypoint_encoder = nn.Linear(num_keypoints, latent_dim)

        ### 4. Cross-Attention ###
        self.cross_attention = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

        ### 5. Temporal Attention Pooling ###
        self.attention_pool = nn.MultiheadAttention(embed_dim=latent_dim, num_heads=4, batch_first=True)

        ### 6. Classification Head ###
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, keypoints, images):
        """
        Args:
            keypoints: Tensor of shape (batch_size, seq_length, num_keypoints)
            images: Tensor of shape (batch_size, seq_length, channels, height, width)
        Returns:
            logits: Tensor of shape (batch_size, num_classes)
        """
        keypoints = keypoints.squeeze(2)
        batch_size, seq_length, _, _, _ = images.size()
        keypoints = keypoints.view(batch_size,seq_length,-1)
        ### 1. Visual Encoder ###
        images = F.interpolate(
                images.view(batch_size * seq_length, images.size(2), images.size(3), images.size(4)),  # Flatten sequence
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        visual_features = self.visual_encoder(images).last_hidden_state  # (batch_size * seq_length, visual_output_dim)
        visual_features = visual_features.mean(dim=1)  
        visual_features = visual_features.view(batch_size, seq_length, -1)  

        ### 2. Perceiver Resampler ###
        visual_tokens = self.perceiver_resampler(inputs=visual_features).last_hidden_state  # (batch_size, num_visual_tokens, latent_dim)

        ### 3. Keypoint Encoder ###
        keypoint_embeddings = self.keypoint_encoder(keypoints)  # (batch_size, seq_length, latent_dim)

        ### 4. Gated Cross-Attention ###
        fused_embeddings, _ = self.cross_attention(query=keypoint_embeddings, key=visual_tokens, value=visual_tokens)

        ### 5. Temporal Attention Pooling ###
        pooled_embedding, _ = self.attention_pool(query=fused_embeddings, key=fused_embeddings, value=fused_embeddings)
        pooled_embedding = pooled_embedding.mean(dim=1)  

        ### 6. Classification ###
        logits = self.classifier(pooled_embedding)
        return logits





class DeadliftDataset(Dataset):
    def __init__(self, folder_paths, seq_length, pose_detection_model, barbell_detection_model, transform=None):
        self.data = []
        self.labels = []
        self.seq_length = seq_length  # Number of frames per sequence
        self.transform = transform
        self.pose_detection_model = pose_detection_model
        self.barbell_detection_model = barbell_detection_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
            pose_result = self.pose_detection_model(frame)

            # Barbell detection
            barbell_result = self.barbell_detection_model(frame)

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
                continue  
            keypoints = keypoints.to(self.device)
            combined_vector = torch.cat((keypoints, torch.tensor(barbell_coords, device = self.device).unsqueeze(0).unsqueeze(0)), dim = 1)
            frame_buffer.append((frame, combined_vector))

            if len(frame_buffer) == self.seq_length:
                images, keypoints_sequence = zip(*frame_buffer)  
                images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])  
                keypoints_sequence = torch.stack(keypoints_sequence, dim=0)   
                self.data.append((images, keypoints_sequence))
                self.labels.append(label_idx)
                frame_buffer.pop(0)  

        cap.release()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images, keypoints = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            images = self.transform(images)

        return keypoints, images, label
