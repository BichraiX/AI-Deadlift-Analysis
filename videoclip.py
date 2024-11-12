import torch.nn as nn

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
