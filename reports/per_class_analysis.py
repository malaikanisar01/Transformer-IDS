import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

REPORT_DIR = BASE_DIR / "reports"
MAPPING_FILE = (
    BASE_DIR
    / "Dataset"
    / "processed"
    / "splits"
    / "label_mapping.csv"
)


# ============================================================
# FILES
# ============================================================

MODEL_FILES = {
    "Random Forest": REPORT_DIR / "random_forest_results.json",
    "LSTM": REPORT_DIR / "lstm_results.json",
    "Transformer": REPORT_DIR / "transformer_results.json",
}


# ============================================================
# LOAD LABEL MAPPING
# ============================================================

print("=" * 70)
print("PER-CLASS MODEL ANALYSIS")
print("=" * 70)

print("\nLoading label mapping...")

mapping_df = pd.read_csv(
    MAPPING_FILE,
    encoding="utf-8-sig"
)

mapping_df["Label_ID"] = mapping_df["Label_ID"].astype(int)

label_mapping = dict(
    zip(
        mapping_df["Label_ID"],
        mapping_df["Label"]
    )
)

print(
    f"Loaded {len(label_mapping)} class labels."
)


# ============================================================
# LOAD MODEL RESULTS
# ============================================================

model_results = {}

for model_name, file_path in MODEL_FILES.items():

    print(
        f"\nLoading {model_name}..."
    )

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as f:

        model_results[model_name] = json.load(f)

    print("Loaded successfully.")


# ============================================================
# CREATE PER-CLASS TABLE
# ============================================================

print("\nCreating per-class comparison...")

rows = []

for class_id in range(15):

    class_name = label_mapping.get(
        class_id,
        f"Unknown Class {class_id}"
    )

    row = {
        "Class_ID": class_id,
        "Attack": class_name,
    }

    for model_name, results in model_results.items():

        report = results.get(
            "classification_report",
            {}
        )

        class_key = str(class_id)

        if class_key in report:

            metrics = report[class_key]

            row[
                f"{model_name}_Precision"
            ] = metrics.get(
                "precision",
                0.0
            )

            row[
                f"{model_name}_Recall"
            ] = metrics.get(
                "recall",
                0.0
            )

            row[
                f"{model_name}_F1"
            ] = metrics.get(
                "f1-score",
                0.0
            )

            row[
                f"{model_name}_Support"
            ] = metrics.get(
                "support",
                0
            )

        else:

            row[
                f"{model_name}_Precision"
            ] = 0.0

            row[
                f"{model_name}_Recall"
            ] = 0.0

            row[
                f"{model_name}_F1"
            ] = 0.0

            row[
                f"{model_name}_Support"
            ] = 0

    rows.append(row)


df = pd.DataFrame(rows)


# ============================================================
# DISPLAY TABLE
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS F1 COMPARISON")
print("=" * 70)

display_columns = [
    "Class_ID",
    "Attack",
    "Random Forest_F1",
    "LSTM_F1",
    "Transformer_F1",
]

print(
    df[display_columns].to_string(
        index=False
    )
)


# ============================================================
# SAVE COMPLETE CSV
# ============================================================

csv_path = (
    REPORT_DIR /
    "per_class_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"\nComplete CSV saved:\n{csv_path}"
)


# ============================================================
# SAVE F1-ONLY CSV
# ============================================================

f1_df = df[
    [
        "Class_ID",
        "Attack",
        "Random Forest_F1",
        "LSTM_F1",
        "Transformer_F1",
    ]
]

f1_csv_path = (
    REPORT_DIR /
    "per_class_f1_comparison.csv"
)

f1_df.to_csv(
    f1_csv_path,
    index=False,
    encoding="utf-8-sig"
)

print(
    f"F1 CSV saved:\n{f1_csv_path}"
)


# ============================================================
# MARKDOWN REPORT
# ============================================================

markdown_path = (
    REPORT_DIR /
    "per_class_analysis.md"
)

with open(
    markdown_path,
    "w",
    encoding="utf-8"
) as f:

    f.write("# Per-Class Model Analysis\n\n")

    f.write(
        "This report compares Random Forest, LSTM, "
        "and Transformer performance for each "
        "CICIDS2017 attack class.\n\n"
    )

    f.write(
        "## F1-Score Comparison\n\n"
    )

    f.write(
        f1_df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    f.write("\n\n")

    f.write(
        "## Detailed Metrics\n\n"
    )

    f.write(
        df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

print(
    f"Markdown report saved:\n{markdown_path}"
)


# ============================================================
# F1 COMPARISON GRAPH
# ============================================================

print("\nCreating F1 comparison graph...")

plt.figure(figsize=(15, 8))

x = range(len(df))

plt.plot(
    x,
    df["Random Forest_F1"],
    marker="o",
    label="Random Forest"
)

plt.plot(
    x,
    df["LSTM_F1"],
    marker="o",
    label="LSTM"
)

plt.plot(
    x,
    df["Transformer_F1"],
    marker="o",
    label="Transformer"
)

plt.xticks(
    list(x),
    df["Attack"],
    rotation=45,
    ha="right"
)

plt.xlabel("Attack Class")
plt.ylabel("F1-Score")

plt.title(
    "Per-Class F1-Score Comparison"
)

plt.ylim(
    0,
    1.05
)

plt.grid(
    True,
    alpha=0.3
)

plt.legend()

plt.tight_layout()

graph_path = (
    REPORT_DIR /
    "per_class_f1_comparison.png"
)

plt.savefig(
    graph_path,
    dpi=300
)

plt.close()

print(
    f"Graph saved:\n{graph_path}"
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PER-CLASS ANALYSIS COMPLETED")
print("=" * 70)

print("\nGenerated files:")

print("1.", csv_path)
print("2.", f1_csv_path)
print("3.", markdown_path)
print("4.", graph_path)

print("\n" + "=" * 70)