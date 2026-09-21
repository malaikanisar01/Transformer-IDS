import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = BASE_DIR / "Dataset" / "processed" / "sequences"
REPORT_DIR = BASE_DIR / "reports"
MODEL_DIR = BASE_DIR / "models" / "saved_models"

REPORT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cpu")

NUM_CLASSES = 15
INPUT_SIZE = 78
SEQUENCE_LENGTH = 32

HIDDEN_SIZE_1 = 64
HIDDEN_SIZE_2 = 32

BATCH_SIZE = 256
EPOCHS = 5

LEARNING_RATE = 0.001

RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


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
            np.asarray(self.X[index], dtype=np.float32)
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
# METRICS
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)

            outputs = model(X_batch)

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                y_batch.cpu().numpy()
            )

    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            all_labels,
            all_predictions,
            average="macro",
            zero_division=0,
        )
    )

    return {
        "accuracy": float(accuracy),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "labels": all_labels,
        "predictions": all_predictions,
    }


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("TRANSFORMER IDS - LSTM MODEL")
print("=" * 70)

print("\nDevice:", DEVICE)

print("\nLoading datasets...")

train_dataset = SequenceDataset(
    SEQUENCE_DIR / "X_train.npy",
    SEQUENCE_DIR / "y_train.npy",
)

validation_dataset = SequenceDataset(
    SEQUENCE_DIR / "X_validation.npy",
    SEQUENCE_DIR / "y_validation.npy",
)

test_dataset = SequenceDataset(
    SEQUENCE_DIR / "X_test.npy",
    SEQUENCE_DIR / "y_test.npy",
)

print("Train sequences     :", len(train_dataset))
print("Validation sequences:", len(validation_dataset))
print("Test sequences      :", len(test_dataset))


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# CLASS WEIGHTS
# ============================================================

print("\nCalculating class weights...")

y_train = np.asarray(
    np.load(
        SEQUENCE_DIR / "y_train.npy",
        mmap_mode="r"
    )
)

class_counts = np.bincount(
    y_train,
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for class_id, count in enumerate(class_counts):

    print(
        f"Class {class_id:2d}: {count:,}"
    )


# Use inverse-frequency weighting only for
# classes that actually occur in training.

class_weights = np.zeros(
    NUM_CLASSES,
    dtype=np.float32
)

total_samples = len(y_train)

for class_id in range(NUM_CLASSES):

    if class_counts[class_id] > 0:

        class_weights[class_id] = (
            total_samples /
            (
                NUM_CLASSES *
                class_counts[class_id]
            )
        )

# Normalize weights for numerical stability.

nonzero = class_weights > 0

class_weights[nonzero] /= np.mean(
    class_weights[nonzero]
)

# Classes absent from training receive zero
# because there are no training examples for them.

print("\nClass weights:")

for class_id, weight in enumerate(class_weights):

    print(
        f"Class {class_id:2d}: {weight:.4f}"
    )

class_weights_tensor = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=DEVICE,
)


# ============================================================
# MODEL
# ============================================================

model = LSTMClassifier().to(DEVICE)

print("\nModel architecture:\n")

print(model)

criterion = nn.CrossEntropyLoss(
    weight=class_weights_tensor
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("STARTING LSTM TRAINING")
print("=" * 70)

best_macro_f1 = -1.0
best_epoch = 0
training_history = []

training_start = time.time()

for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_start = time.time()

    running_loss = 0.0
    total_batches = 0

    for batch_index, (X_batch, y_batch) in enumerate(
        train_loader,
        start=1
    ):

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(X_batch)

        loss = criterion(
            outputs,
            y_batch
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        running_loss += loss.item()

        total_batches += 1

        if batch_index % 100 == 0:

            print(
                f"Epoch {epoch}/{EPOCHS} | "
                f"Batch {batch_index}/{len(train_loader)} | "
                f"Loss {loss.item():.4f}"
            )

    average_loss = (
        running_loss / total_batches
    )

    validation_metrics = evaluate(
        model,
        validation_loader
    )

    epoch_time = (
        time.time() - epoch_start
    )

    print("\n" + "-" * 70)

    print(
        f"Epoch {epoch}/{EPOCHS}"
    )

    print(
        f"Loss          : {average_loss:.4f}"
    )

    print(
        f"Val Accuracy   : "
        f"{validation_metrics['accuracy']:.4f}"
    )

    print(
        f"Val Macro F1   : "
        f"{validation_metrics['macro_f1']:.4f}"
    )

    print(
        f"Epoch Time     : "
        f"{epoch_time:.2f} sec"
    )

    print("-" * 70)

    training_history.append({
        "epoch": epoch,
        "loss": float(average_loss),
        "validation_accuracy":
            validation_metrics["accuracy"],
        "validation_macro_f1":
            validation_metrics["macro_f1"],
        "epoch_time_seconds":
            float(epoch_time),
    })

    if (
        validation_metrics["macro_f1"]
        > best_macro_f1
    ):

        best_macro_f1 = (
            validation_metrics["macro_f1"]
        )

        best_epoch = epoch

        torch.save(
            model.state_dict(),
            MODEL_DIR / "lstm_best.pt"
        )

        print(
            f"\nBest model saved "
            f"(epoch {epoch})."
        )


training_time = (
    time.time() - training_start
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\nLoading best LSTM model...")

model.load_state_dict(
    torch.load(
        MODEL_DIR / "lstm_best.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)


# ============================================================
# FINAL TEST
# ============================================================

print("\n" + "=" * 70)
print("FINAL LSTM TEST")
print("=" * 70)

test_metrics = evaluate(
    model,
    test_loader
)

y_test = test_metrics["labels"]
y_pred = test_metrics["predictions"]


precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)


print(
    f"\nAccuracy          : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Macro Precision   : "
    f"{test_metrics['macro_precision']:.4f}"
)

print(
    f"Macro Recall      : "
    f"{test_metrics['macro_recall']:.4f}"
)

print(
    f"Macro F1          : "
    f"{test_metrics['macro_f1']:.4f}"
)

print(
    f"Weighted Precision: "
    f"{precision_weighted:.4f}"
)

print(
    f"Weighted Recall   : "
    f"{recall_weighted:.4f}"
)

print(
    f"Weighted F1       : "
    f"{f1_weighted:.4f}"
)

print(
    f"Training Time     : "
    f"{training_time:.2f} sec"
)

print(
    f"Best Epoch        : "
    f"{best_epoch}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:\n")

classification_text = classification_report(
    y_test,
    y_pred,
    zero_division=0,
)

print(classification_text)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(
    np.unique(
        np.concatenate(
            [y_test, y_pred]
        )
    )
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels,
)

print("\nConfusion Matrix:\n")

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "LSTM",

    "input_shape": [
        SEQUENCE_LENGTH,
        INPUT_SIZE
    ],

    "sequence_length":
        SEQUENCE_LENGTH,

    "features":
        INPUT_SIZE,

    "hidden_size_1":
        HIDDEN_SIZE_1,

    "hidden_size_2":
        HIDDEN_SIZE_2,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "best_epoch":
        best_epoch,

    "training_time_seconds":
        float(training_time),

    "accuracy":
        test_metrics["accuracy"],

    "macro_precision":
        test_metrics["macro_precision"],

    "macro_recall":
        test_metrics["macro_recall"],

    "macro_f1":
        test_metrics["macro_f1"],

    "weighted_precision":
        float(precision_weighted),

    "weighted_recall":
        float(recall_weighted),

    "weighted_f1":
        float(f1_weighted),

    "classification_report":
        classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        ),

    "confusion_matrix":
        cm.tolist(),

    "training_history":
        training_history,
}


report_path = (
    REPORT_DIR /
    "lstm_results.json"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        results,
        f,
        indent=4
    )


print(
    "\nResults saved to:"
)

print(report_path)

print(
    "\nModel saved to:"
)

print(
    MODEL_DIR /
    "lstm_best.pt"
)

print(
    "\n" + "=" * 70
)

print(
    "LSTM TRAINING COMPLETED"
)

print(
    "=" * 70
)