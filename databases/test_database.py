from database import (
    initialize_database,
    save_prediction,
    get_predictions
)


print("=" * 60)
print("DATABASE TEST")
print("=" * 60)


# Step 1: Initialize database
initialize_database()

print("Database initialized successfully.")


# Step 2: Insert test prediction
prediction_id = save_prediction(
    timestamp="2026-09-20T13:30:00",
    predicted_class_id=2,
    predicted_class_name="DDoS",
    confidence=0.999260,
    time_steps=32,
    features_per_step=78
)

print(f"Prediction saved with ID: {prediction_id}")


# Step 3: Retrieve predictions
predictions = get_predictions(limit=10)

print("\nSaved predictions:")

for prediction in predictions:
    print(prediction)


print("=" * 60)
print("DATABASE TEST PASSED")
print("=" * 60)