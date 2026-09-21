import numpy as np
import requests
import csv
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_PATH = PROJECT_ROOT / "reports" / "api_test_results.csv"


# ============================================================
# TEST DATA
# ============================================================

X_test = np.load(
    PROJECT_ROOT / "Dataset" / "processed" / "sequences" / "X_test.npy",
    mmap_mode="r"
)

y_test = np.load(
    PROJECT_ROOT / "Dataset" / "processed" / "sequences" / "y_test.npy"
)


# ============================================================
# LABEL MAPPING
# ============================================================

LABEL_MAPPING = {
    0: "BENIGN",
    1: "Bot",
    2: "DDoS",
    3: "DoS GoldenEye",
    4: "DoS Hulk",
    5: "DoS Slowhttptest",
    6: "DoS slowloris",
    7: "FTP-Patator",
    8: "Heartbleed",
    9: "Infiltration",
    10: "PortScan",
    11: "SSH-Patator",
    12: "Web Attack - Brute Force",
    13: "Web Attack - Sql Injection",
    14: "Web Attack - XSS"
}


# ============================================================
# FIND ONE SAMPLE PER AVAILABLE CLASS
# ============================================================

class_indices = {}

for index, class_id in enumerate(y_test):

    class_id = int(class_id)

    if class_id not in class_indices:
        class_indices[class_id] = index


# ============================================================
# API TEST
# ============================================================

results = []

for class_id in sorted(class_indices):

    index = class_indices[class_id]

    sequence = X_test[index].tolist()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"sequence": sequence}
    )

    response.raise_for_status()

    data = response.json()

    prediction = data["prediction"]

    predicted_id = int(prediction["class_id"])
    confidence = float(prediction["confidence"])

    correct = predicted_id == class_id

    results.append({
        "sequence_index": index,
        "actual_class_id": class_id,
        "actual_class": LABEL_MAPPING[class_id],
        "predicted_class_id": predicted_id,
        "predicted_class": LABEL_MAPPING.get(
            predicted_id,
            "Unknown"
        ),
        "correct": correct,
        "confidence": confidence
    })


# ============================================================
# SAVE CSV REPORT
# ============================================================

REPORT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

with open(
    REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "sequence_index",
        "actual_class_id",
        "actual_class",
        "predicted_class_id",
        "predicted_class",
        "correct",
        "confidence"
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(results)


# ============================================================
# SUMMARY
# ============================================================

total = len(results)
passed = sum(
    result["correct"]
    for result in results
)

failed = total - passed
pass_rate = (passed / total) * 100


print("=" * 70)
print("API TEST REPORT")
print("=" * 70)

for result in results:

    status = "PASS" if result["correct"] else "FAIL"

    print(
        f"{status} | "
        f"{result['actual_class']} -> "
        f"{result['predicted_class']} | "
        f"Confidence: {result['confidence']:.6f}"
    )

print("=" * 70)
print(f"Total classes tested : {total}")
print(f"Passed               : {passed}")
print(f"Failed               : {failed}")
print(f"Pass rate            : {pass_rate:.2f}%")
print(f"Report saved to      : {REPORT_PATH}")
print("=" * 70)