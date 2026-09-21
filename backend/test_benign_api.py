import requests
import numpy as np


API_URL = "http://127.0.0.1:8000"


# ============================================================
# LOAD TEST SEQUENCE
# ============================================================

X_test = np.load(
    "Dataset/processed/sequences/X_test.npy"
)

y_test = np.load(
    "Dataset/processed/sequences/y_test.npy"
)


# ============================================================
# FIND BENIGN SEQUENCE
# ============================================================

benign_index = np.where(y_test == 0)[0][0]

sequence = X_test[benign_index]


print(
    "Selected test index:",
    benign_index
)

print(
    "Actual class:",
    y_test[benign_index]
)

print(
    "Sequence shape:",
    sequence.shape
)


# ============================================================
# SEND REQUEST
# ============================================================

payload = {
    "sequence": sequence.tolist()
}


response = requests.post(
    f"{API_URL}/predict",
    json=payload,
    timeout=30
)


# ============================================================
# DISPLAY RESULT
# ============================================================

print(
    "\nHTTP Status:",
    response.status_code
)

print(
    "Response:"
)

print(
    response.json()
)