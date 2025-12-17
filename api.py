
import pandas as pd
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import sys
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
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
#   2 = Long (>=400s)
# =====================================================

from fastapi.middleware.cors import CORSMiddleware

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

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# =====================================================
# Load Model
# =====================================================
MODEL_PATH = os.getenv("MODEL_PATH", "solar_flare_model.pkl")
DATASET_PATH = os.getenv("DATASET_PATH", "Solar Flares from RHESSI Mission -10.csv")

# Global DataFrame
df = None

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

# Load Dataset & Pre-calculate Stats
EDA_STATS = None
try:
    if os.path.exists(DATASET_PATH):
        df = pd.read_csv(DATASET_PATH)
        print(f"✅ Dataset loaded successfully: {len(df)} rows")
        
        # Pre-calc stats for speed
        print("⏳ Pre-calculating statistics...")
        desc = df.describe().to_dict()
        nulls = df.isnull().sum().to_dict()
        EDA_STATS = {
            "columns": list(df.columns),
            "shape": {"rows": df.shape[0], "cols": df.shape[1]},
            "null_counts": nulls,
            "statistics": desc
        }
        print("✅ Statistics ready")
        
    else:
        print(f"⚠️ Dataset not found at {DATASET_PATH}")
        df = None
except Exception as e:
    print(f"❌ Error loading dataset: {e}")



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

# =====================================================
# Endpoints
# =====================================================

@app.get("/api-info", tags=["Info"])
def api_info():
    """Welcome endpoint with API information"""
    return {
        "message": "🌟 Solar Flare Prediction API is running!",
        "endpoints": {
            "GET /": "App Interface",
            "GET /api-info": "This welcome message",
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



@app.get("/random-sample", tags=["Data"])
def get_random_sample():
    """Get a random row from the loaded dataset"""
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    # Sample 1 random row
    random_row = df.sample(n=1).iloc[0].to_dict()
    
    # Handle NaN values for JSON serialization
    clean_row = {k: (None if pd.isna(v) else v) for k, v in random_row.items()}
    
    return {
        "dataset_total_rows": len(df),
        "row": clean_row
    }

@app.get("/eda-stats", tags=["Data"])
def get_eda_stats():
    """Get basic descriptive statistics of the dataset"""
    if EDA_STATS is None:
        raise HTTPException(status_code=404, detail="Dataset stats not available")
    
    return EDA_STATS

@app.get("/analysis", tags=["Pages"])
async def read_analysis():
    return FileResponse('frontend/analysis.html')

@app.get("/columns", tags=["Analysis"])
def get_columns():
    """Get list of numeric columns for plotting"""
    if df is None:
         raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    # Return only numeric columns for safety
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return {"columns": numeric_cols}

@app.get("/plot/correlation", tags=["Analysis"])
def get_correlation_plot():
    """Generate a correlation heatmap"""
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    plt.figure(figsize=(10, 8))
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, cmap='magma', cbar=True)
    plt.title("Feature Correlation Matrix", color='white')
    
    # Style for dark mode
    fig = plt.gcf()
    fig.patch.set_facecolor('#0F0F0F')
    ax = plt.gca()
    ax.set_facecolor('#0F0F0F')
    ax.tick_params(colors='white', which='both')
    
    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    return Response(content=buf.getvalue(), media_type="image/png")
    # return Response(content=b"", media_type="image/png")

@app.get("/plot/distribution", tags=["Analysis"])
def get_distribution_plot(column: str = "duration.s"):
    """Generate a distribution plot for a specific column"""
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
    
    if column not in df.columns:
        raise HTTPException(status_code=400, detail=f"Column {column} not found")

    plt.figure(figsize=(8, 5))
    sns.histplot(df[column], kde=True, color='#4CAF50', bins=30)
    plt.title(f"Distribution of {column}", color='white')
    plt.xlabel(column, color='white')
    plt.ylabel("Count", color='white')
    
    # Style
    fig = plt.gcf()
    fig.patch.set_facecolor('#0F0F0F')
    ax = plt.gca()
    ax.set_facecolor('#111')
    ax.tick_params(colors='gray', which='both')
    for spine in ax.spines.values():
        spine.set_color('#333')

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    return Response(content=buf.getvalue(), media_type="image/png")
    # return Response(content=b"", media_type="image/png")

@app.get("/plot/scatter", tags=["Analysis"])
def get_scatter_plot(x: str = "duration.s", y: str = "total.counts"):
    """Generate a scatter plot between two columns"""
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")

    if x not in df.columns or y not in df.columns:
         raise HTTPException(status_code=400, detail="Column not found")

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x=x, y=y, color='#ED64A6', alpha=0.6, s=15)
    plt.title(f"{x} vs {y}", color='white')
    plt.xlabel(x, color='white')
    plt.ylabel(y, color='white')
    
    # Style
    fig = plt.gcf()
    fig.patch.set_facecolor('#0F0F0F')
    ax = plt.gca()
    ax.set_facecolor('#111')
    ax.tick_params(colors='gray', which='both')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    return Response(content=buf.getvalue(), media_type="image/png")
    # return Response(content=b"", media_type="image/png")

@app.get("/plot/null-distribution", tags=["Analysis"])
def get_null_distribution_plot():
    """Generate a bar plot of null values per column"""
    if df is None:
        raise HTTPException(status_code=404, detail="Dataset not loaded")
        
    null_counts = df.isnull().sum()
    # Filter only columns with nulls for cleaner plot, or show all if few
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    
    if null_counts.empty:
         # Create a blank plot saying "No Missing Values"
        plt.figure(figsize=(8, 2))
        plt.text(0.5, 0.5, "No Missing Values Found", ha='center', va='center', color='white', fontsize=14)
        plt.axis('off')
    else:
        plt.figure(figsize=(10, 6))
        sns.barplot(x=null_counts.values, y=null_counts.index, palette="viridis")
        plt.title("Missing Values per Column", color='white')
        plt.xlabel("Count of Nulls", color='white')
        
    # Style
    fig = plt.gcf()
    fig.patch.set_facecolor('#0F0F0F')
    ax = plt.gca()
    ax.set_facecolor('#111')
    ax.tick_params(colors='gray', which='both')
    for spine in ax.spines.values():
        spine.set_color('#333')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    plt.close()
    
    return Response(content=buf.getvalue(), media_type="image/png")

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

# Mount Frontend at Root (Must be last to avoid overriding API routes)
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# =====================================================
# Run with: uvicorn api:app --host 0.0.0.0 --port 8000
# =====================================================