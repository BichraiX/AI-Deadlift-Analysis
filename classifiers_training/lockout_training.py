from perceiver_based_classifier import DeadliftMovementClassifier

# Model Configuration
num_keypoints = 36  # e.g., 18 joints with x, y coordinates
pretrained_visual_model_name = "google/vit-base-patch16-224-in21k"  # Example: Vision Transformer
latent_dim = 256
num_visual_tokens = 16
num_classes = 2  # Good or Bad movement

# Instantiate Model
model = DeadliftMovementClassifier(num_keypoints, pretrained_visual_model_name, latent_dim, num_visual_tokens, num_classes)