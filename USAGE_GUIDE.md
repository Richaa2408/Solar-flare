# 🌟 Solar Flare Prediction App - User Guide

This guide explains how to run the SHEP (Stellar High-Energy Pulse) Prediction System locally on your machine.

---

## 🚀 Quick Start (Windows)

1.  **Double-click** the `run_local.bat` file in this folder.
2.  Wait for the server to start (you'll see `Uvicorn running on http://0.0.0.0:8000`).
3.  Open your browser to: **[http://localhost:8000](http://localhost:8000)**

That's it!

---

## 💻 Manual Start (Command Line)

If you prefer using the terminal:

1.  **Install Expectations**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **Start the Server**:
    ```bash
    uvicorn api:app --host 0.0.0.0 --port 8000
    ```

3.  **Open App**:
    Go to **[http://localhost:8000](http://localhost:8000)**

---

## 🧭 Navigation

- **Predictor** (`/`):
    - Load a random sample from the dataset.
    - Run the AI model to predict solar flare duration.
- **Analysis** (`/analysis.html`):
    - Visualize feature distributions.
    - Explore correlations and scatter plots properly.

## 🔧 Troubleshooting

- **Dataset Not Found**: Ensure `Solar Flares from RHESSI Mission -10.csv` is in the same folder as `api.py`.
- **Model Not Found**: Ensure `solar_flare_model.pkl` is present.
- **Port In Use**: If 8000 is taken, run with `--port 8001`.
