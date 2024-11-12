import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.io import read_video

import open_clip

import os

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
