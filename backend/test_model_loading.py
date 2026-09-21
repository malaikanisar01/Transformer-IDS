import sys
from pathlib import Path

import torch


# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Allow importing from models folder
sys.path.insert(0, str(PROJECT_ROOT))

from models.transformer_model import TransformerIDS


# Model path
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "transformer_best.pt"
)


print("=" * 60)
print("TRANSFORMER MODEL LOADING TEST")
print("=" * 60)

print("Model path:")
print(MODEL_PATH)

print("File exists:", MODEL_PATH.exists())


# Create exact same architecture
model = TransformerIDS(
    input_dim=78,
    num_classes=15,
    d_model=64,
    n_heads=4,
    num_layers=2,
    dropout=0.1,
    max_seq_len=32
)


# Load trained weights
checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)


# Handle either plain state_dict or checkpoint dictionary
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)


model.eval()


# Test input
sample = torch.randn(1, 32, 78)


with torch.no_grad():
    output = model(sample)


prediction = torch.argmax(
    output,
    dim=1
).item()


print("\nModel loaded successfully!")
print("Input shape :", sample.shape)
print("Output shape:", output.shape)
print("Predicted class ID:", prediction)

print("=" * 60)
print("MODEL LOADING TEST PASSED")
print("=" * 60)