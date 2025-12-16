import pandas as pd
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional, Union
import os

# =====================================================
# Solar Flare Duration Classifier API
# =====================================================
# This API allows you to classify solar flare durations
# by passing a single row of data in the same format
# as the training data.
#
# Classes:
#   0 = Short (<200s)
#   1 = Medium (<400s) 
#   2 = Long (>=400s)
# =====================================================

app = FastAPI(
    title="Solar Flare Duration Classifier",
    description="""
    ## Solar Flare Prediction API
    
    This API classifies solar flare durations based on RHESSI mission data.
    
    ### How to Use
    
    1. **Check expected features**: `GET /features`
    2. **Get example input**: `GET /example`  
    3. **Make prediction**: `POST /predict` with your data
    
    You can pass your **entire row** from the dataset - the API will automatically 
    extract the relevant numeric features needed for prediction.
    
    ### Example Request (full row from dataset)
    ```python
    import requests
    import pandas as pd
    
    # Load your dataframe
    df = pd.read_csv("your_solar_flare_data.csv")
    
    # Get any row and convert to dict
    row = df.iloc[0].to_dict()
    
    # Make prediction - API extracts relevant features automatically
    response = requests.post("http://localhost:8000/predict", json={"row": row})
    print(response.json())
    ```
    """,
    version="1.0.0"
)

# =====================================================
# Load Model
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "solar_flare_model.pkl")

try:
    data = joblib.load(MODEL_PATH)
    model = data["model"]
    model_features = data["features"]
    print(f"✅ Model loaded successfully from: {MODEL_PATH}")
    print(f"📊 Expected features: {model_features}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    model_features = []

# =====================================================
# Schemas
# =====================================================
class PredictionRequest(BaseModel):
    """Single row prediction request - accepts full row from dataset"""
    row: Dict[str, Any] = Field(
        ...,
        description="A full row from your dataframe as a dictionary. The API will automatically extract the required numeric features.",
        example={
            "flare": 2021213,
            "start.date": "12-02-02",
            "start.time": "21:29:56",
            "peak": "21:33:38",
            "end": "21:41:48",
            "duration.s": 712,
            "peak.c/s": 136,
            "total.counts": 167304,
            "energy.kev": "Dec-25",
            "x.pos.asec": 592,
            "y.pos.asec": -358,
            "radial": 692,
            "active.region.ar": 0,
            "flag.1": "A1",
            "flag.2": "P1",
            "flag.3": None,
            "flag.4": None,
            "flag.5": None
        }
    )

class PredictionResponse(BaseModel):
    prediction_class: int
    prediction_label: str
    features_used: Dict[str, float]
    confidence_note: str = "Prediction based on LightGBM classifier trained on RHESSI solar flare data"

class BatchPredictionRequest(BaseModel):
    """Multiple rows prediction request - accepts full rows from dataset"""
    rows: List[Dict[str, Any]] = Field(
        ...,
        description="List of full row dictionaries from your dataframe. The API will extract required features automatically."
    )

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    expected_features: List[str]

# =====================================================
# Helper Functions
# =====================================================
def extract_features(row_dict: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract only the required numeric features from a full dataset row.
    Handles type conversion and missing values.
    """
    extracted = {}
    missing_features = []
    
    for feature in model_features:
        if feature in row_dict:
            value = row_dict[feature]
            # Handle None/NaN values
            if value is None or (isinstance(value, float) and pd.isna(value)):
                extracted[feature] = 0.0
            else:
                try:
                    extracted[feature] = float(value)
                except (ValueError, TypeError):
                    extracted[feature] = 0.0
        else:
            missing_features.append(feature)
            extracted[feature] = 0.0
    
    return extracted, missing_features

# =====================================================
# Endpoints
# =====================================================

@app.get("/", tags=["Info"])
def home():
    """Welcome endpoint with API information"""
    return {
        "message": "🌟 Solar Flare Prediction API is running!",
        "endpoints": {
            "GET /": "This welcome message",
            "GET /health": "Check API and model status",
            "GET /features": "Get list of expected features",
            "GET /example": "Get an example input for prediction",
            "POST /predict": "Make a single prediction",
            "POST /predict/batch": "Make batch predictions"
        },
        "documentation": "/docs"
    }

@app.get("/health", response_model=HealthResponse, tags=["Info"])
def health():
    """Health check endpoint to verify the API and model status"""
    return {
        "status": "healthy" if model else "unhealthy",
        "model_loaded": model is not None,
        "expected_features": model_features
    }

@app.get("/features", tags=["Info"])
def get_features():
    """Get the list of features expected by the model"""
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    return {
        "features": model_features,
        "count": len(model_features),
        "description": "These are the feature names your input data should contain"
    }

@app.get("/example", tags=["Info"])
def get_example():
    """Get an example input that can be used for prediction - shows a full dataset row"""
    # This is an example of a FULL row from your dataset
    example_full_row = {
        "flare": 2021213,
        "start.date": "12-02-02",
        "start.time": "21:29:56",
        "peak": "21:33:38",
        "end": "21:41:48",
        "duration.s": 712,
        "peak.c/s": 136,
        "total.counts": 167304,
        "energy.kev": "Dec-25",
        "x.pos.asec": 592,
        "y.pos.asec": -358,
        "radial": 692,
        "active.region.ar": 0,
        "flag.1": "A1",
        "flag.2": "P1",
        "flag.3": None,
        "flag.4": None,
        "flag.5": None
    }
    
    # Show which features will be extracted
    extracted_features = {k: v for k, v in example_full_row.items() if k in model_features}
    
    return {
        "info": "You can pass your ENTIRE row from the dataset. The API will automatically extract the required features.",
        "required_features": model_features,
        "example_full_row": example_full_row,
        "features_that_will_be_extracted": extracted_features,
        "example_request": {
            "row": example_full_row
        },
        "python_code": '''
import requests
import pandas as pd

# Load your dataframe
df = pd.read_csv("Solar Flares from RHESSI Mission -10.csv")

# Get any row and convert to dict
row = df.iloc[0].to_dict()

# Make prediction - API extracts relevant features automatically!
response = requests.post("http://localhost:8000/predict", json={"row": row})
print(response.json())
'''
    }

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
def predict(request: PredictionRequest):

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    # Extract only the required features from the full row
    extracted, missing = extract_features(request.row)
    
    # Create DataFrame with extracted features
    df = pd.DataFrame([extracted])
    
    # Ensure correct column order
    try:
        df_processed = df.reindex(columns=model_features, fill_value=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing features: {str(e)}")

    # Make Prediction
    prediction_idx = model.predict(df_processed)[0]
    
    # Map index back to class name
    class_mapping = {0: "Short (<200s)", 1: "Medium (<400s)", 2: "Long (>=400s)"}
    result = class_mapping.get(int(prediction_idx), "Unknown")

    return {
        "prediction_class": int(prediction_idx),
        "prediction_label": result,
        "features_used": extracted,
        "confidence_note": "Prediction based on LightGBM classifier trained on RHESSI solar flare data"
    }

@app.post("/predict/batch", tags=["Predictions"])
def predict_batch(request: BatchPredictionRequest):

    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if not request.rows:
        raise HTTPException(status_code=400, detail="No rows provided")
    
    # Extract features from each row
    extracted_rows = []
    for row in request.rows:
        extracted, _ = extract_features(row)
        extracted_rows.append(extracted)
    
    # Create DataFrame from extracted features
    df = pd.DataFrame(extracted_rows)
    
    try:
        df_processed = df.reindex(columns=model_features, fill_value=0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing features: {str(e)}")
    
    # Make predictions
    predictions = model.predict(df_processed)
    
    class_mapping = {0: "Short (<200s)", 1: "Medium (<400s)", 2: "Long (>=400s)"}
    
    results = [
        {
            "row_index": i,
            "prediction_class": int(pred),
            "prediction_label": class_mapping.get(int(pred), "Unknown")
        }
        for i, pred in enumerate(predictions)
    ]
    
    return {
        "predictions": results,
        "total_rows": len(results),
        "features_used": model_features
    }

# =====================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
# =====================================================