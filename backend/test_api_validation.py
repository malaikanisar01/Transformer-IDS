import requests


# ============================================================
# TEST 1: Wrong number of time steps
# ============================================================

wrong_sequence = [[0.0] * 78 for _ in range(31)]

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": wrong_sequence}
)

print("=" * 60)
print("TEST 1: WRONG TIME STEPS")
print("=" * 60)
print("Status Code:", response.status_code)
print("Response:", response.json())


# ============================================================
# TEST 2: Wrong number of features
# ============================================================

wrong_sequence = [[0.0] * 77 for _ in range(32)]

response = requests.post(
    "http://127.0.0.1:8000/predict",
    json={"sequence": wrong_sequence}
)

print("\n" + "=" * 60)
print("TEST 2: WRONG FEATURES")
print("=" * 60)
print("Status Code:", response.status_code)
print("Response:", response.json())