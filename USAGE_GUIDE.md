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

### Step 2: Load the Docker Image
#### 1. Download the image
Download the Docker image from the link below and place it in your project directory:

https://drive.google.com/file/d/1GhRKX1ij6ctEpLD52IdzlT8p211vySDf/view?usp=sharing

Open a terminal in the folder containing your files and run:
```bash
docker load -i solar-flare-api.tar
```

You should see: `Loaded image: solar-flare-api:latest`

---

### Step 3: Start the API Server

Run this command:

```bash
docker run -p 8000:8000 solar-flare-api
```

✅ **Success looks like:**
```
✅ Model loaded successfully from: /app/solar_flare_model.pkl
📊 Expected features: ['flare', 'peak.c/s', 'total.counts', 'x.pos.asec', 'y.pos.asec', 'radial', 'active.region.ar']
INFO:     Uvicorn running on http://0.0.0.0:8000
```

> ⚠️ **Keep this terminal open!** The API runs as long as this terminal is open.

---

### Step 4: Test the API (Browser)

Open your browser and go to:

| URL | What it shows |
|-----|---------------|
| http://localhost:8000 | Welcome message |
| http://localhost:8000/health | API status |
| http://localhost:8000/docs | Interactive documentation |
| http://localhost:8000/features | List of input features |

---

### Step 5: Run the Python Client

Open a **new terminal** (keep the API running in the first one).

Install dependencies:
```bash
pip install requests pandas
```

Run the client:
```bash
python client_example.py
```

✅ **Expected output:**
```
Solar Flare API Client Example (Random CSV Row)

1. Checking API health...
   Status: healthy
   Model loaded: True
   Expected features: ['flare', 'peak.c/s', 'total.counts', ...]

2. Predicting using random CSV row...
   Prediction class: 2
   Prediction label: Long (>=400s)
   Features used: {'flare': 2021213.0, 'peak.c/s': 136.0, ...}

Done!
```

---

## 📊 Understanding the Output

The model classifies solar flare duration into 3 categories:

| Class | Label | Duration |
|-------|-------|----------|
| 0 | Short | Less than 200 seconds |
| 1 | Medium | 200-399 seconds |
| 2 | Long | 400+ seconds |

---

## 💡 Quick Reference Commands

```bash
# Start the API
docker run -p 8000:8000 solar-flare-api

# Stop the API
# Press Ctrl+C in the terminal running Docker

# Run client script
python client_example.py

# Check if API is running
curl http://localhost:8000/health
```

---

## 🔧 Troubleshooting

- **Dataset Not Found**: Ensure `Solar Flares from RHESSI Mission -10.csv` is in the same folder as `api.py`.
- **Model Not Found**: Ensure `solar_flare_model.pkl` is present.
- **Port In Use**: If 8000 is taken, run with `--port 8001`.
