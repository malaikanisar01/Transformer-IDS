import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

SEQUENCE_DIR = BASE_DIR / "Dataset" / "processed" / "sequences"
REPORT_DIR = BASE_DIR / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

# Keep CPU training practical.
# Maximum samples per class used for RF training.
MAX_SAMPLES_PER_CLASS = 10000

N_ESTIMATORS = 150


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("TRANSFORMER IDS - RANDOM FOREST BASELINE")
print("=" * 70)

print("\nLoading sequence data...")

X_train = np.load(SEQUENCE_DIR / "X_train.npy", mmap_mode="r")
y_train = np.load(SEQUENCE_DIR / "y_train.npy", mmap_mode="r")

X_test = np.load(SEQUENCE_DIR / "X_test.npy", mmap_mode="r")
y_test = np.load(SEQUENCE_DIR / "y_test.npy", mmap_mode="r")

print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"X_test : {X_test.shape}")
print(f"y_test : {y_test.shape}")


# ============================================================
# LAST TIMESTEP FEATURES
# ============================================================

print("\nConverting temporal sequences to RF features...")

X_train_last = np.asarray(X_train[:, -1, :], dtype=np.float32)
X_test_last = np.asarray(X_test[:, -1, :], dtype=np.float32)

y_train_np = np.asarray(y_train)
y_test_np = np.asarray(y_test)

print(f"RF training features: {X_train_last.shape}")
print(f"RF testing features : {X_test_last.shape}")


# ============================================================
# BALANCED TRAINING SAMPLE
# ============================================================

print("\nCreating balanced training sample...")

rng = np.random.default_rng(RANDOM_STATE)

selected_indices = []

unique_classes = np.unique(y_train_np)

for class_id in unique_classes:

    class_indices = np.flatnonzero(y_train_np == class_id)

    if len(class_indices) > MAX_SAMPLES_PER_CLASS:
        class_indices = rng.choice(
            class_indices,
            size=MAX_SAMPLES_PER_CLASS,
            replace=False,
        )

    selected_indices.append(class_indices)

selected_indices = np.concatenate(selected_indices)

rng.shuffle(selected_indices)

X_train_rf = X_train_last[selected_indices]
y_train_rf = y_train_np[selected_indices]

print(f"\nOriginal training samples: {len(y_train_np):,}")
print(f"RF training samples      : {len(y_train_rf):,}")

print("\nTraining class distribution:")

unique, counts = np.unique(y_train_rf, return_counts=True)

for class_id, count in zip(unique, counts):
    print(f"Class {class_id:2d}: {count:,}")


# ============================================================
# RANDOM FOREST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)

model = RandomForestClassifier(
    n_estimators=N_ESTIMATORS,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample",
    max_features="sqrt",
    max_depth=None,
)

start_time = time.time()

model.fit(X_train_rf, y_train_rf)

training_time = time.time() - start_time

print(f"\nTraining completed in {training_time:.2f} seconds.")


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

start_time = time.time()

y_pred = model.predict(X_test_last)

prediction_time = time.time() - start_time

print(f"Prediction completed in {prediction_time:.2f} seconds.")


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test_np, y_pred)

precision_macro, recall_macro, f1_macro, _ = (
    precision_recall_fscore_support(
        y_test_np,
        y_pred,
        average="macro",
        zero_division=0,
    )
)

precision_weighted, recall_weighted, f1_weighted, _ = (
    precision_recall_fscore_support(
        y_test_np,
        y_pred,
        average="weighted",
        zero_division=0,
    )
)

print("\n" + "=" * 70)
print("RANDOM FOREST RESULTS")
print("=" * 70)

print(f"Accuracy          : {accuracy:.4f}")
print(f"Macro Precision   : {precision_macro:.4f}")
print(f"Macro Recall      : {recall_macro:.4f}")
print(f"Macro F1          : {f1_macro:.4f}")
print(f"Weighted Precision: {precision_weighted:.4f}")
print(f"Weighted Recall   : {recall_weighted:.4f}")
print(f"Weighted F1       : {f1_weighted:.4f}")
print(f"Training Time     : {training_time:.2f} sec")
print(f"Prediction Time   : {prediction_time:.2f} sec")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test_np,
    y_pred,
    zero_division=0,
    output_dict=True,
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test_np,
        y_pred,
        zero_division=0,
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

labels = sorted(np.unique(np.concatenate([y_test_np, y_pred])))

cm = confusion_matrix(
    y_test_np,
    y_pred,
    labels=labels,
)

print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {
    "model": "Random Forest",
    "input_representation": "last_timestep_of_sequence",
    "sequence_length": 32,
    "features": 78,
    "random_state": RANDOM_STATE,
    "max_samples_per_class": MAX_SAMPLES_PER_CLASS,
    "n_estimators": N_ESTIMATORS,
    "training_samples": int(len(y_train_rf)),
    "test_samples": int(len(y_test_np)),
    "accuracy": float(accuracy),
    "macro_precision": float(precision_macro),
    "macro_recall": float(recall_macro),
    "macro_f1": float(f1_macro),
    "weighted_precision": float(precision_weighted),
    "weighted_recall": float(recall_weighted),
    "weighted_f1": float(f1_weighted),
    "training_time_seconds": float(training_time),
    "prediction_time_seconds": float(prediction_time),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
}

report_path = REPORT_DIR / "random_forest_results.json"

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to:")
print(report_path)


# ============================================================
# SAVE MODEL
# ============================================================

import joblib

MODEL_DIR = BASE_DIR / "models" / "saved_models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

model_path = MODEL_DIR / "random_forest.pkl"

joblib.dump(model, model_path)

print("\nModel saved to:")
print(model_path)

print("\n" + "=" * 70)
print("RANDOM FOREST TRAINING COMPLETED")
print("=" * 70)