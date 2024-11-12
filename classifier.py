import open_clip
import torch.nn as nn
import cv2
import torch
from torchvision import transforms

# Load the OpenCLIP model and tokenizer
model, preprocess = open_clip.create_model_and_transforms("ViT-B/32", pretrained="laion2b_s34b_b88k")
tokenizer = open_clip.get_tokenizer("ViT-B/32")

# Freeze some layers if needed (optional, helps with training stability)
for param in model.parameters():
    param.requires_grad = False



# Define transformation for frames to match CLIP input size (e.g., 224x224)
clip_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# Function to extract and preprocess frames
def extract_frames(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, total_frames // num_frames)

    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = clip_transform(frame)
            frames.append(frame)
        if len(frames) == num_frames:
            break

    cap.release()
    return torch.stack(frames)  # Shape: [num_frames, 3, 224, 224]
