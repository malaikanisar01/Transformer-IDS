import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from transformer_model import TransformerIDS


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SEQUENCE_DIR = os.path.join(
    BASE_DIR,
    "Dataset",
    "processed",
    "sequences"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "saved_models"
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# Dataset parameters
INPUT_DIM = 78
NUM_CLASSES = 15
SEQUENCE_LENGTH = 32

# Training parameters
BATCH_SIZE = 256
EPOCHS = 5
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")

# Keep CPU workload reasonable
torch.set_num_threads(2)


# ============================================================
# DATASET CLASS
# ============================================================

class SequenceDataset(Dataset):

    def __init__(self, x_path, y_path):

        self.X = np.load(
            x_path,
            mmap_mode="r"
        )

        self.y = np.load(
            y_path,
            mmap_mode="r"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):

        # copy() avoids the read-only NumPy memory-map warning
        x = torch.tensor(
            self.X[index].copy(),
            dtype=torch.float32
        )

        y = torch.tensor(
            int(self.y[index]),
            dtype=torch.long
        )

        return x, y


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRANSFORMER IDS - MODEL TRAINING")
print("=" * 70)

print("\nLoading sequence datasets...")

train_dataset = SequenceDataset(
    os.path.join(SEQUENCE_DIR, "X_train.npy"),
    os.path.join(SEQUENCE_DIR, "y_train.npy")
)

validation_dataset = SequenceDataset(
    os.path.join(SEQUENCE_DIR, "X_validation.npy"),
    os.path.join(SEQUENCE_DIR, "y_validation.npy")
)

test_dataset = SequenceDataset(
    os.path.join(SEQUENCE_DIR, "X_test.npy"),
    os.path.join(SEQUENCE_DIR, "y_test.npy")
)

print("Train sequences      :", len(train_dataset))
print("Validation sequences :", len(validation_dataset))
print("Test sequences       :", len(test_dataset))


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

y_train = np.load(
    os.path.join(SEQUENCE_DIR, "y_train.npy"),
    mmap_mode="r"
)

class_counts = np.bincount(
    np.asarray(y_train),
    minlength=NUM_CLASSES
)

print("\nTraining class distribution:")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id:2d}: {count}")


# Use square-root inverse frequency.
# This is less aggressive than the previous LSTM weighting.
class_weights = np.zeros(NUM_CLASSES, dtype=np.float32)

total_samples = len(y_train)

for class_id in range(NUM_CLASSES):

    if class_counts[class_id] > 0:

        class_weights[class_id] = np.sqrt(
            total_samples / class_counts[class_id]
        )

# Normalize weights so average non-zero weight is approximately 1
non_zero = class_weights > 0

class_weights[non_zero] = (
    class_weights[non_zero]
    / class_weights[non_zero].mean()
)

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("\nClass weights:")

for class_id, weight in enumerate(class_weights):
    print(f"Class {class_id:2d}: {weight:.4f}")


# ============================================================
# MODEL
# ============================================================

print("\nCreating Transformer model...")

model = TransformerIDS(
    input_dim=INPUT_DIM,
    num_classes=NUM_CLASSES,
    d_model=64,
    n_heads=4,
    num_layers=2,
    dropout=0.1,
    max_seq_len=SEQUENCE_LENGTH
).to(DEVICE)

print(model)


# ============================================================
# LOSS + OPTIMIZER
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            outputs = model(X)

            loss = criterion(
                outputs,
                y
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y.cpu().numpy()
            )

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    return (
        total_loss / len(loader),
        accuracy,
        macro_f1
    )


# ============================================================
# TRAINING
# ============================================================

print("\nStarting Transformer training...")
print("Device:", DEVICE)
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("=" * 70)

best_macro_f1 = -1
best_epoch = 0

training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    epoch_start = time.time()

    model.train()

    total_loss = 0.0

    for batch_index, (X, y) in enumerate(train_loader):

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(
            outputs,
            y
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (batch_index + 1) % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index + 1}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    train_loss = total_loss / len(train_loader)

    validation_loss, validation_accuracy, validation_macro_f1 = evaluate(
        model,
        validation_loader
    )

    epoch_time = time.time() - epoch_start

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Train Loss      : {train_loss:.4f}"
    )

    print(
        f"Validation Loss  : {validation_loss:.4f}"
    )

    print(
        f"Validation Acc   : {validation_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: {validation_macro_f1:.4f}"
    )

    print(
        f"Epoch Time       : {epoch_time:.2f} sec"
    )

    print("-" * 70)

    # Save best model
    if validation_macro_f1 > best_macro_f1:

        best_macro_f1 = validation_macro_f1
        best_epoch = epoch

        torch.save(
            model.state_dict(),
            os.path.join(
                MODEL_DIR,
                "transformer_best.pt"
            )
        )

        print(
            f"Best Transformer model saved "
            f"(Epoch {epoch})"
        )


training_time = time.time() - training_start


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best Transformer model...")

model.load_state_dict(
    torch.load(
        os.path.join(
            MODEL_DIR,
            "transformer_best.pt"
        ),
        map_location=DEVICE
    )
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

print("\nRunning final test evaluation...")

model.eval()

all_predictions = []
all_labels = []

prediction_start = time.time()

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(DEVICE)

        outputs = model(X)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            y.numpy()
        )

prediction_time = time.time() - prediction_start


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_precision = precision_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_recall = recall_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_precision = precision_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_recall = recall_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("TRANSFORMER FINAL RESULTS")
print("=" * 70)

print(f"Accuracy           : {accuracy:.4f}")
print(f"Macro Precision    : {macro_precision:.4f}")
print(f"Macro Recall       : {macro_recall:.4f}")
print(f"Macro F1           : {macro_f1:.4f}")
print(f"Weighted Precision : {weighted_precision:.4f}")
print(f"Weighted Recall    : {weighted_recall:.4f}")
print(f"Weighted F1        : {weighted_f1:.4f}")

print(f"\nBest Epoch         : {best_epoch}")
print(f"Best Validation F1 : {best_macro_f1:.4f}")
print(f"Training Time      : {training_time:.2f} sec")
print(f"Prediction Time    : {prediction_time:.2f} sec")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "Transformer",
    "input_dim": INPUT_DIM,
    "sequence_length": SEQUENCE_LENGTH,
    "num_classes": NUM_CLASSES,
    "d_model": 64,
    "n_heads": 4,
    "num_layers": 2,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,

    "best_epoch": best_epoch,
    "best_validation_macro_f1": float(best_macro_f1),

    "accuracy": float(accuracy),

    "macro_precision": float(macro_precision),
    "macro_recall": float(macro_recall),
    "macro_f1": float(macro_f1),

    "weighted_precision": float(weighted_precision),
    "weighted_recall": float(weighted_recall),
    "weighted_f1": float(weighted_f1),

    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time),

    "classification_report": classification_report(
        all_labels,
        all_predictions,
        output_dict=True,
        zero_division=0
    ),

    "confusion_matrix": cm.tolist()
}

results_path = os.path.join(
    REPORT_DIR,
    "transformer_results.json"
)

with open(
    results_path,
    "w"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print("\nResults saved to:")
print(results_path)

print("\nModel saved to:")
print(
    os.path.join(
        MODEL_DIR,
        "transformer_best.pt"
    )
)

print("\n" + "=" * 70)
print("TRANSFORMER TRAINING COMPLETED")
print("=" * 70)
