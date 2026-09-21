import numpy as np
import requests

# Load one real test sequence
X_test = np.load(
    r"Dataset\processed\sequences\X_test.npy",
    mmap_mode="r"
)

sequence = X_test[0].tolist()

# Send sequence to FastAPI
response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": sequence}
)

print("Status Code:", response.status_code)
print("API Response:")
print(response.json())