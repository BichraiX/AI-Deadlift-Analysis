import torch
import torch.nn as nn
from transformers import PerceiverModel, PerceiverConfig, AutoModel
import torch.nn.functional as F

class DeadliftMovementClassifier(nn.Module):
    def __init__(self, num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes):
        super(DeadliftMovementClassifier, self).__init__()

        ### 1. Pretrained Visual Encoder ###
        # Use a pretrained vision model (frozen)
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

        ### 4. Gated Cross-Attention ###
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
        images = images.permute(0, 1, 4, 2, 3)
        keypoints = keypoints.squeeze(2)
        batch_size, seq_length, _, _, _ = images.size()
        keypoints = keypoints.view(batch_size,seq_length,-1)
        ### 1. Visual Encoder ###
        # Flatten sequence for batch processing by the visual encoder
        images = F.interpolate(
                images.view(batch_size * seq_length, images.size(2), images.size(3), images.size(4)),  # Flatten sequence
                size=(224, 224),
                mode='bilinear',
                align_corners=False
            )
        visual_features = self.visual_encoder(images).last_hidden_state  # (batch_size * seq_length, visual_output_dim)
        visual_features = visual_features.mean(dim=1)  # Global average pooling over spatial dimensions
        visual_features = visual_features.view(batch_size, seq_length, -1)  # Reshape back to sequence

        ### 2. Perceiver Resampler ###
        visual_tokens = self.perceiver_resampler(inputs=visual_features).last_hidden_state  # (batch_size, num_visual_tokens, latent_dim)

        ### 3. Keypoint Encoder ###
        keypoint_embeddings = self.keypoint_encoder(keypoints)  # (batch_size, seq_length, latent_dim)

        ### 4. Gated Cross-Attention ###
        # Queries: keypoints; Keys/Values: visual tokens
        fused_embeddings, _ = self.cross_attention(query=keypoint_embeddings, key=visual_tokens, value=visual_tokens)

        ### 5. Temporal Attention Pooling ###
        # Input sequence is fused_embeddings
        pooled_embedding, _ = self.attention_pool(query=fused_embeddings, key=fused_embeddings, value=fused_embeddings)
        pooled_embedding = pooled_embedding.mean(dim=1)  # Average pooling over temporal dimension

        ### 6. Classification ###
        logits = self.classifier(pooled_embedding)
        return logits



