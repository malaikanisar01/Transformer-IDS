import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = BASE_DIR / "Dataset" / "processed" / "sequences"
MODEL_PATH = BASE_DIR / "models" / "saved_models" / "lstm_best.pt"
RESULT_PATH = BASE_DIR / "reports" / "lstm_results.json"

DEVICE = torch.device("cpu")

INPUT_SIZE = 78
NUM_CLASSES = 15
SEQUENCE_LENGTH = 32

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32

BATCH_SIZE = 256


# ============================================================
# DATASET
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, X_path, y_path):

        self.X = np.load(X_path, mmap_mode="r")
        self.y = np.load(y_path, mmap_mode="r")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        x = torch.from_numpy(
            np.asarray(
                self.X[index],
                dtype=np.float32
            ).copy()
        )

        y = torch.tensor(
            int(self.y[index]),
            dtype=torch.long
        )

        return x, y


# ============================================================
# LSTM MODEL
# ============================================================

class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size=INPUT_SIZE,
        hidden_size_1=HIDDEN_SIZE_1,
        hidden_size_2=HIDDEN_SIZE_2,
        num_classes=NUM_CLASSES,
    ):

        super().__init__()

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size_1,
            batch_first=True,
        )

        self.dropout1 = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=hidden_size_1,
            hidden_size=hidden_size_2,
            batch_first=True,
        )

        self.dropout2 = nn.Dropout(0.3)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size_2, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):

        x, _ = self.lstm1(x)

        x = self.dropout1(x)

        x, _ = self.lstm2(x)

        x = x[:, -1, :]

        x = self.dropout2(x)

        return self.classifier(x)


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("LSTM INFERENCE TIME MEASUREMENT")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading test dataset...")

test_dataset = SequenceDataset(
    SEQUENCE_DIR / "X_test.npy",
    SEQUENCE_DIR / "y_test.npy",
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("Test sequences:", len(test_dataset))


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading saved LSTM model...")

model = LSTMClassifier().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)

model.eval()

print("Model loaded successfully.")


# ============================================================
# WARM-UP
# ============================================================

print("\nPerforming warm-up inference...")

with torch.no_grad():

    for batch_index, (X_batch, _) in enumerate(test_loader):

        X_batch = X_batch.to(DEVICE)

        _ = model(X_batch)

        if batch_index >= 2:
            break


# ============================================================
# ACTUAL INFERENCE
# ============================================================

print("\nMeasuring inference time...")

prediction_start = time.perf_counter()

all_predictions = []

with torch.no_grad():

    for X_batch, _ in test_loader:

        X_batch = X_batch.to(DEVICE)

        outputs = model(X_batch)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

prediction_time = (
    time.perf_counter() -
    prediction_start
)


# ============================================================
# RESULTS
# ============================================================

num_predictions = len(all_predictions)

average_time_per_sequence = (
    prediction_time /
    num_predictions
)

sequences_per_second = (
    num_predictions /
    prediction_time
)

print("\n" + "=" * 70)
print("INFERENCE RESULTS")
print("=" * 70)

print(
    f"\nTotal predictions     : "
    f"{num_predictions:,}"
)

print(
    f"Total inference time  : "
    f"{prediction_time:.4f} sec"
)

print(
    f"Average per sequence  : "
    f"{average_time_per_sequence * 1000:.4f} ms"
)

print(
    f"Sequences per second  : "
    f"{sequences_per_second:.2f}"
)


# ============================================================
# UPDATE JSON
# ============================================================

print("\nUpdating lstm_results.json...")

with open(
    RESULT_PATH,
    "r",
    encoding="utf-8"
) as f:

    results = json.load(f)


results["prediction_time_seconds"] = float(
    prediction_time
)

results["average_inference_time_ms"] = float(
    average_time_per_sequence * 1000
)

results["inference_sequences_per_second"] = float(
    sequences_per_second
)


with open(
    RESULT_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nUpdated successfully:")

print(RESULT_PATH)

print("\n" + "=" * 70)
print("LSTM INFERENCE MEASUREMENT COMPLETED")
print("=" * 70)