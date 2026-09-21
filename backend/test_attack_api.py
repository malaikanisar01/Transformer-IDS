import numpy as np
import requests

# Load test data
X_test = np.load(
    r"Dataset\processed\sequences\X_test.npy",
    mmap_mode="r"
)

y_test = np.load(
    r"Dataset\processed\sequences\y_test.npy"
)

# First attack sequence
attack_index = np.where(y_test != 0)[0][0]

sequence = X_test[attack_index].tolist()
actual_class = int(y_test[attack_index])

print("Test sequence index:", attack_index)
print("Actual class ID:", actual_class)
print("Actual class: DDoS")

# Send to API
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": sequence}
)

print("\nStatus Code:", response.status_code)
print("API Response:")
print(response.json())