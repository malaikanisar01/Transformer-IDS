from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from datetime import datetime
from pathlib import Path
import sys

import torch

from databases.database import (
    initialize_database,
    save_prediction,
    get_predictions
)


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT TRANSFORMER MODEL
# ============================================================

from models.transformer_model import TransformerIDS


# ============================================================
# MODEL CONFIGURATION
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "saved_models"
    / "transformer_best.pt"
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
# LOAD TRANSFORMER
# ============================================================

model = TransformerIDS(
    input_dim=78,
    num_classes=15,
    d_model=64,
    n_heads=4,
    num_layers=2,
    dropout=0.1,
    max_seq_len=32
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu",
    weights_only=True
)

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()


# ============================================================
# CREATE FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Transformer IDS API",
    description="Network Intrusion Detection System API",
    version="1.0.0"
)

initialize_database()

# ============================================================
# REQUEST MODEL
# ============================================================

class PredictionRequest(BaseModel):

    sequence: list[list[float]] = Field(
        ...,
        description="Network traffic sequence containing exactly 32 time steps and 78 features."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "Transformer-Based Network Intrusion Detection System",
        "status": "running",
        "model": "Transformer",
        "message": "Transformer IDS API is working"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": True,
        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================
@app.post("/predict")
def predict(request: PredictionRequest):

    # Validate number of time steps
    if len(request.sequence) != 32:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Sequence must contain exactly 32 time steps.",
                "received_time_steps": len(request.sequence),
                "expected_time_steps": 32
            }
        )

    # Validate number of features
    for index, row in enumerate(request.sequence):
        if len(row) != 78:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": (
                        f"Time step {index} must contain "
                        f"exactly 78 features."
                    ),
                    "time_step": index,
                    "received_features": len(row),
                    "expected_features": 78
                }
            )

    # Convert input to tensor
    try:
        input_tensor = torch.tensor(
            request.sequence,
            dtype=torch.float32
        ).unsqueeze(0)

        # Model prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)

            predicted_class = int(
                torch.argmax(probabilities, dim=1).item()
            )

            confidence = float(
                probabilities[0, predicted_class].item()
            )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

    # Save prediction to database
    timestamp = datetime.now().isoformat()

    prediction_id = save_prediction(
        timestamp=timestamp,
        predicted_class_id=predicted_class,
        predicted_class_name=LABEL_MAPPING.get(
            predicted_class,
            "Unknown"
        ),
        confidence=confidence,
        time_steps=32,
        features_per_step=78
    )

    # Return prediction
    return {
        "success": True,
        "prediction": {
            "class_id": predicted_class,
            "class_name": LABEL_MAPPING.get(
                predicted_class,
                "Unknown"
            ),
            "confidence": round(
                confidence,
                6
            )
        },
        "input": {
            "time_steps": 32,
            "features_per_step": 78
        },
        "database": {
            "prediction_id": prediction_id,
            "saved": True
        },
        "timestamp": timestamp
    }


@app.get("/predictions")
def prediction_history(limit: int = 100):
    """
    Return recent prediction history from the database.
    """

    if limit < 1 or limit > 1000:
        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 1000."
        )

    predictions = get_predictions(limit)

    return {
        "success": True,
        "count": len(predictions),
        "predictions": predictions
    }


    