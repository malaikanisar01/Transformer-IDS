import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORTS_DIR = BASE_DIR / "reports"
MAPPING_FILE = BASE_DIR / "Dataset" / "processed" / "splits" / "label_mapping.csv"


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

import pandas as pd

mapping_df = pd.read_csv(MAPPING_FILE)

label_names = {}

for _, row in mapping_df.iterrows():
    label_names[int(row["Label_ID"])] = str(row["Label"])


# ============================================================
# ALL 15 CLASSES
# ============================================================

NUM_CLASSES = 15

class_names = [
    label_names.get(i, f"Class {i}")
    for i in range(NUM_CLASSES)
]


# Short names for plot readability
short_names = [
    "BENIGN",
    "Bot",
    "DDoS",
    "DoS GoldenEye",
    "DoS Hulk",
    "DoS Slowhttptest",
    "DoS slowloris",
    "FTP-Patator",
    "Heartbleed",
    "Infiltration",
    "PortScan",
    "SSH-Patator",
    "Web Brute Force",
    "Web SQL Injection",
    "Web XSS"
]


# ============================================================
# LOAD MODEL RESULT
# ============================================================

models = {
    "Random Forest": REPORTS_DIR / "random_forest_results.json",
    "LSTM": REPORTS_DIR / "lstm_results.json",
    "Transformer": REPORTS_DIR / "transformer_results.json"
}


# ============================================================
# FUNCTION: PAD CONFUSION MATRIX TO 15 x 15
# ============================================================

def prepare_matrix(raw_matrix, labels):
    """
    Converts a confusion matrix into a full 15 x 15 matrix.

    Some classes are absent from the test set, so their rows/
    columns may not exist in the original matrix.
    """

    raw_matrix = np.array(raw_matrix)

    full_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=int
    )

    labels = [int(x) for x in labels]

    for i, actual_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):

            if actual_label < NUM_CLASSES and predicted_label < NUM_CLASSES:
                full_matrix[actual_label, predicted_label] = raw_matrix[i, j]

    return full_matrix


# ============================================================
# PLOT RAW CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(matrix, model_name, normalized=False):

    if normalized:

        row_sums = matrix.sum(axis=1, keepdims=True)

        normalized_matrix = np.divide(
            matrix,
            row_sums,
            out=np.zeros_like(matrix, dtype=float),
            where=row_sums != 0
        )

        display_matrix = normalized_matrix
        title = f"{model_name} - Normalized Confusion Matrix"

    else:

        display_matrix = matrix
        title = f"{model_name} - Confusion Matrix"


    plt.figure(figsize=(15, 12))

    plt.imshow(display_matrix, interpolation="nearest")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")

    plt.xticks(
        range(NUM_CLASSES),
        short_names,
        rotation=90
    )

    plt.yticks(
        range(NUM_CLASSES),
        short_names
    )

    plt.colorbar()

    # Add values inside cells
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):

            value = display_matrix[i, j]

            if normalized:
                text = f"{value:.2f}"
            else:
                text = str(int(value))

            if value > display_matrix.max() * 0.5:
                text_color = "white"
            else:
                text_color = "black"

            plt.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                color=text_color,
                fontsize=7
            )

    plt.tight_layout()

    filename = (
        model_name.lower().replace(" ", "_")
        + "_confusion_matrix"
    )

    if normalized:
        filename += "_normalized"

    filename += ".png"

    output_path = REPORTS_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {output_path}")


# ============================================================
# MAIN
# ============================================================

print("=" * 60)
print("TRANSFORMER IDS - CONFUSION MATRIX ANALYSIS")
print("=" * 60)


for model_name, json_file in models.items():

    print(f"\nProcessing: {model_name}")

    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    raw_matrix = results["confusion_matrix"]

    # JSON may contain explicit labels
    labels = results.get(
        "confusion_matrix_labels",
        list(range(len(raw_matrix)))
    )

    matrix = prepare_matrix(
        raw_matrix,
        labels
    )

    print("Matrix shape:", matrix.shape)
    print("Total samples:", matrix.sum())

    # Raw matrix
    plot_confusion_matrix(
        matrix,
        model_name,
        normalized=False
    )

    # Normalized matrix
    plot_confusion_matrix(
        matrix,
        model_name,
        normalized=True
    )


print("\n" + "=" * 60)
print("CONFUSION MATRIX ANALYSIS COMPLETED")
print("=" * 60)