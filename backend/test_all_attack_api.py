import numpy as np
import requests
from collections import defaultdict

# ------------------------------------------------------------
# Load test sequences and labels
# ------------------------------------------------------------

X_test = np.load(
    r"Dataset\processed\sequences\X_test.npy",
    mmap_mode="r"
)

y_test = np.load(
    r"Dataset\processed\sequences\y_test.npy"
)

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

# ------------------------------------------------------------
# Find one real test sequence for each available class
# ------------------------------------------------------------

class_indices = {}

for index, class_id in enumerate(y_test):
    class_id = int(class_id)

    if class_id not in class_indices:
        class_indices[class_id] = index

# ------------------------------------------------------------
# Test API
# ------------------------------------------------------------

results = []

for class_id in sorted(class_indices):

    index = class_indices[class_id]

    sequence = X_test[index].tolist()

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"sequence": sequence}
    )

    data = response.json()

    predicted_id = data["prediction"]["class_id"]
    confidence = data["prediction"]["confidence"]

    correct = predicted_id == class_id

    results.append(correct)

    status = "PASS" if correct else "FAIL"

    print(
        f"{status} | "
        f"Index: {index} | "
        f"Actual: {class_id} ({LABEL_MAPPING[class_id]}) | "
        f"Predicted: {predicted_id} "
        f"({LABEL_MAPPING.get(predicted_id, 'Unknown')}) | "
        f"Confidence: {confidence}"
    )

# ------------------------------------------------------------
# Summary
# ------------------------------------------------------------

print("\n" + "=" * 80)

total = len(results)
passed = sum(results)

print(f"API TEST SUMMARY")
print(f"Total classes tested : {total}")
print(f"Passed               : {passed}")
print(f"Failed               : {total - passed}")
print(f"Pass rate            : {(passed / total) * 100:.2f}%")

print("=" * 80)