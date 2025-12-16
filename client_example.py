

import requests
import pandas as pd
import random

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
API_URL = "http://localhost:8000"
CSV_PATH = "Solar Flares from RHESSI Mission -10.csv"


# --------------------------------------------------
# API HELPERS
# --------------------------------------------------
def check_health():
    """Check if the API is running and model is loaded"""
    response = requests.get(f"{API_URL}/health")
    return response.json()


def get_required_features():
    """Get the list of features the model expects"""
    response = requests.get(f"{API_URL}/features")
    return response.json()


def predict_single(row_dict: dict):
    """
    Make a prediction for a single row.
    """
    response = requests.post(
        f"{API_URL}/predict",
        json={"row": row_dict}
    )
    return response.json()


def predict_batch(rows: list):
    """
    Make predictions for multiple rows at once.
    """
    response = requests.post(
        f"{API_URL}/predict/batch",
        json={"rows": rows}
    )
    return response.json()


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------
def load_random_row_from_csv(csv_path: str) -> dict:
    """
    Load the CSV and return a random row as a dictionary.
    """
    df = pd.read_csv(csv_path)

    if df.empty:
        raise ValueError("CSV file is empty!")

    random_row = df.sample(n=1).iloc[0]

    # Convert NaN to None for JSON compatibility
    row_dict = random_row.where(pd.notnull(random_row), None).to_dict()
    return row_dict


# ============================================================
# EXAMPLE USAGE
# ============================================================
if __name__ == "__main__":
    print("=" * 50)
    print("Solar Flare API Client Example (Random CSV Row)")
    print("=" * 50)

    # 1. Check API health
    print("\n1. Checking API health...")
    try:
        health = check_health()
        print(f"   Status: {health['status']}")
        print(f"   Model loaded: {health['model_loaded']}")
        print(f"   Expected features: {health['expected_features']}")
    except requests.exceptions.ConnectionError:
        print("   ERROR: Cannot connect to API!")
        print("   Start the API using:")
        print("   uvicorn api:app --host 0.0.0.0 --port 8000")
        exit(1)

    try:
        random_row = load_random_row_from_csv(CSV_PATH)
        print("   Random row loaded successfully")
    except Exception as e:
        print(f"   ERROR loading CSV: {e}")
        exit(1)

    # 3. Predict using random CSV row
    print("\n2. Predicting using random CSV row...")
    result = predict_single(random_row)

    print(f"   Prediction class: {result['prediction_class']}")
    print(f"   Prediction label: {result['prediction_label']}")
    print(f"   Features used: {result['features_used']}")

    print("\n" + "=" * 50)
    print("Done!")
