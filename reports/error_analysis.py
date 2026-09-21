import json
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = BASE_DIR / "reports"
MAPPING_FILE = (
    BASE_DIR
    / "Dataset"
    / "processed"
    / "splits"
    / "label_mapping.csv"
)


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

mapping_df = pd.read_csv(MAPPING_FILE)

label_names = {
    int(row["Label_ID"]): str(row["Label"])
    for _, row in mapping_df.iterrows()
}


# ============================================================
# MODEL FILES
# ============================================================

models = {
    "Random Forest": REPORTS_DIR / "random_forest_results.json",
    "LSTM": REPORTS_DIR / "lstm_results.json",
    "Transformer": REPORTS_DIR / "transformer_results.json",
}


# ============================================================
# CONVERT MATRIX TO FULL 15 x 15
# ============================================================

NUM_CLASSES = 15


def prepare_matrix(raw_matrix, labels):

    raw_matrix = np.array(raw_matrix)

    full_matrix = np.zeros(
        (NUM_CLASSES, NUM_CLASSES),
        dtype=int
    )

    labels = [int(x) for x in labels]

    for i, actual_label in enumerate(labels):
        for j, predicted_label in enumerate(labels):

            if (
                actual_label < NUM_CLASSES
                and predicted_label < NUM_CLASSES
            ):
                full_matrix[
                    actual_label,
                    predicted_label
                ] = raw_matrix[i, j]

    return full_matrix


# ============================================================
# ANALYZE MODEL ERRORS
# ============================================================

all_error_rows = []
all_summary_rows = []


for model_name, json_file in models.items():

    print("\n" + "=" * 70)
    print(f"MODEL: {model_name}")
    print("=" * 70)

    with open(json_file, "r", encoding="utf-8") as f:
        results = json.load(f)

    raw_matrix = results["confusion_matrix"]

    labels = results.get(
        "confusion_matrix_labels",
        list(range(len(raw_matrix)))
    )

    matrix = prepare_matrix(
        raw_matrix,
        labels
    )

    total_samples = int(matrix.sum())

    # --------------------------------------------------------
    # Total correct and incorrect
    # --------------------------------------------------------

    correct = int(np.trace(matrix))
    incorrect = total_samples - correct

    error_rate = incorrect / total_samples

    print(f"Total samples : {total_samples:,}")
    print(f"Correct       : {correct:,}")
    print(f"Incorrect     : {incorrect:,}")
    print(f"Error rate    : {error_rate:.4%}")

    all_summary_rows.append({
        "Model": model_name,
        "Total Samples": total_samples,
        "Correct Predictions": correct,
        "Incorrect Predictions": incorrect,
        "Error Rate": error_rate,
        "Accuracy": results["accuracy"],
        "Macro F1": results["macro_f1"]
    })

    # --------------------------------------------------------
    # Per-class errors
    # --------------------------------------------------------

    for actual_class in range(NUM_CLASSES):

        total_actual = int(matrix[actual_class].sum())

        if total_actual == 0:
            continue

        correct_class = int(
            matrix[actual_class, actual_class]
        )

        errors_class = total_actual - correct_class

        error_rate_class = (
            errors_class / total_actual
        )

        # Find most common wrong prediction
        wrong_predictions = matrix[
            actual_class
        ].copy()

        wrong_predictions[
            actual_class
        ] = 0

        predicted_class = int(
            np.argmax(wrong_predictions)
        )

        largest_confusion = int(
            wrong_predictions[predicted_class]
        )

        all_error_rows.append({
            "Model": model_name,
            "Actual Class ID": actual_class,
            "Actual Class": label_names.get(
                actual_class,
                f"Class {actual_class}"
            ),
            "Support": total_actual,
            "Correct": correct_class,
            "Errors": errors_class,
            "Error Rate": error_rate_class,
            "Most Common Wrong Prediction ID": predicted_class,
            "Most Common Wrong Prediction": label_names.get(
                predicted_class,
                f"Class {predicted_class}"
            ),
            "Confusion Count": largest_confusion
        })


# ============================================================
# CREATE DATAFRAMES
# ============================================================

error_df = pd.DataFrame(all_error_rows)
summary_df = pd.DataFrame(all_summary_rows)


# ============================================================
# SORT BY NUMBER OF ERRORS
# ============================================================

error_df = error_df.sort_values(
    by=["Model", "Errors"],
    ascending=[True, False]
)


# ============================================================
# SAVE CSV FILES
# ============================================================

error_csv = REPORTS_DIR / "error_analysis.csv"
summary_csv = REPORTS_DIR / "error_analysis_summary.csv"

error_df.to_csv(
    error_csv,
    index=False
)

summary_df.to_csv(
    summary_csv,
    index=False
)


# ============================================================
# TOP ERROR CASES
# ============================================================

print("\n" + "=" * 70)
print("TOP ERROR CASES")
print("=" * 70)

for model_name in models.keys():

    model_errors = error_df[
        error_df["Model"] == model_name
    ]

    print(f"\n--- {model_name} ---")

    top_rows = model_errors.head(5)

    for _, row in top_rows.iterrows():

        print(
            f"{row['Actual Class']} -> "
            f"{row['Most Common Wrong Prediction']} | "
            f"Errors: {row['Errors']} | "
            f"Support: {row['Support']} | "
            f"Error Rate: {row['Error Rate']:.2%}"
        )


# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("MODEL ERROR SUMMARY")
print("=" * 70)

for _, row in summary_df.iterrows():

    print(
        f"{row['Model']}: "
        f"Accuracy={row['Accuracy']:.4f}, "
        f"Macro F1={row['Macro F1']:.4f}, "
        f"Errors={row['Incorrect Predictions']:,}, "
        f"Error Rate={row['Error Rate']:.2%}"
    )


# ============================================================
# COMPLETED
# ============================================================

print("\n" + "=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print(f"Saved: {error_csv}")
print(f"Saved: {summary_csv}")