import os
import json
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

REPORT_DIR = os.path.join(
    BASE_DIR,
    "reports"
)


# ============================================================
# RESULT FILES
# ============================================================

model_files = {
    "Random Forest": "random_forest_results.json",
    "LSTM": "lstm_results.json",
    "Transformer": "transformer_results.json"
}


# ============================================================
# LOAD RESULTS
# ============================================================

results = {}

print("=" * 70)
print("TRANSFORMER IDS - MODEL COMPARISON")
print("=" * 70)

for model_name, filename in model_files.items():

    filepath = os.path.join(
        REPORT_DIR,
        filename
    )

    print(f"\nLoading {model_name}...")

    with open(filepath, "r") as file:
        results[model_name] = json.load(file)

    print("Loaded successfully.")


# ============================================================
# BUILD COMPARISON TABLE
# ============================================================

comparison = []

for model_name, result in results.items():

    comparison.append({
        "Model": model_name,

        "Accuracy": result.get(
            "accuracy",
            0
        ),

        "Macro Precision": result.get(
            "macro_precision",
            0
        ),

        "Macro Recall": result.get(
            "macro_recall",
            0
        ),

        "Macro F1": result.get(
            "macro_f1",
            0
        ),

        "Weighted Precision": result.get(
            "weighted_precision",
            0
        ),

        "Weighted Recall": result.get(
            "weighted_recall",
            0
        ),

        "Weighted F1": result.get(
            "weighted_f1",
            0
        ),

        "Training Time (sec)": result.get(
            "training_time_seconds",
            0
        ),

        "Prediction Time (sec)": result.get(
            "prediction_time_seconds",
            0
        )
    })


df = pd.DataFrame(comparison)


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("MODEL PERFORMANCE COMPARISON")
print("=" * 70)

print(
    df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# SAVE CSV
# ============================================================

csv_path = os.path.join(
    REPORT_DIR,
    "model_comparison.csv"
)

df.to_csv(
    csv_path,
    index=False
)

print("\nComparison CSV saved:")
print(csv_path)


# ============================================================
# SAVE JSON
# ============================================================

json_path = os.path.join(
    REPORT_DIR,
    "model_comparison.json"
)

with open(
    json_path,
    "w"
) as file:

    json.dump(
        comparison,
        file,
        indent=4
    )

print("\nComparison JSON saved:")
print(json_path)


# ============================================================
# SAVE MARKDOWN REPORT
# ============================================================

md_path = os.path.join(
    REPORT_DIR,
    "model_comparison.md"
)

with open(
    md_path,
    "w",
    encoding="utf-8"
) as file:

    file.write("# Transformer IDS - Model Comparison\n\n")

    file.write(
        "This report compares the three trained models "
        "using the same test dataset.\n\n"
    )

    file.write(
        "## Overall Performance\n\n"
    )

    file.write(
        df.to_markdown(
            index=False,
            floatfmt=".4f"
        )
    )

    file.write("\n\n")

    file.write(
        "## Evaluation Metrics\n\n"
    )

    file.write(
        "- Accuracy measures overall classification correctness.\n"
        "- Macro Precision gives equal importance to each class.\n"
        "- Macro Recall measures detection capability across classes.\n"
        "- Macro F1 balances precision and recall across classes.\n"
        "- Weighted F1 accounts for class frequency.\n\n"
    )

    file.write(
        "Because the CICIDS2017 dataset is highly imbalanced, "
        "Macro F1 and Macro Recall are particularly important "
        "when evaluating minority attack classes.\n\n"
    )

    file.write(
        "## Training Time\n\n"
    )

    file.write(
        "Training time is reported in seconds and depends on "
        "the available CPU hardware and software environment.\n"
    )

print("\nMarkdown report saved:")
print(md_path)


# ============================================================
# GRAPH 1 — ACCURACY
# ============================================================

plt.figure()

plt.bar(
    df["Model"],
    df["Accuracy"]
)

plt.title(
    "Model Accuracy Comparison"
)

plt.ylabel(
    "Accuracy"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

accuracy_graph = os.path.join(
    REPORT_DIR,
    "accuracy_comparison.png"
)

plt.savefig(
    accuracy_graph,
    dpi=300
)

plt.close()

print("\nAccuracy graph saved:")
print(accuracy_graph)


# ============================================================
# GRAPH 2 — MACRO F1
# ============================================================

plt.figure()

plt.bar(
    df["Model"],
    df["Macro F1"]
)

plt.title(
    "Macro F1 Comparison"
)

plt.ylabel(
    "Macro F1"
)

plt.ylim(
    0,
    1
)

plt.xticks(
    rotation=15
)

plt.tight_layout()

f1_graph = os.path.join(
    REPORT_DIR,
    "macro_f1_comparison.png"
)

plt.savefig(
    f1_graph,
    dpi=300
)

plt.close()

print("\nMacro F1 graph saved:")
print(f1_graph)


# ============================================================
# GRAPH 3 — PRECISION / RECALL / F1
# ============================================================

metrics = [
    "Macro Precision",
    "Macro Recall",
    "Macro F1"
]

x = range(len(df["Model"]))

width = 0.25

plt.figure()

for i, metric in enumerate(metrics):

    values = df[metric]

    positions = [
        value + (i - 1) * width
        for value in x
    ]

    plt.bar(
        positions,
        values,
        width=width,
        label=metric
    )

plt.xticks(
    list(x),
    df["Model"],
    rotation=15
)

plt.ylabel(
    "Score"
)

plt.title(
    "Macro Precision, Recall and F1"
)

plt.ylim(
    0,
    1
)

plt.legend()

plt.tight_layout()

metrics_graph = os.path.join(
    REPORT_DIR,
    "macro_metrics_comparison.png"
)

plt.savefig(
    metrics_graph,
    dpi=300
)

plt.close()

print("\nMacro metrics graph saved:")
print(metrics_graph)


# ============================================================
# FINISHED
# ============================================================

print("\n")
print("=" * 70)
print("MODEL COMPARISON COMPLETED SUCCESSFULLY")
print("=" * 70)