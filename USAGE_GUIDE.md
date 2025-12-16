# 🌟 Solar Flare Prediction API - Usage Guide

A beginner-friendly guide to run the Solar Flare Duration Classifier using Docker.

---

## 📁 What You Need

Before starting, make sure you have these 3 files in the same folder:

| File | Description |
|------|-------------|
| `solar-flare-api.tar` | The Docker image (pre-built) |
| `Solar Flares from RHESSI Mission -10.csv` | The dataset |
| `client_example.py` | Python script to call the API |

---

## 🚀 Step-by-Step Instructions

### Step 1: Install Docker

If you don't have Docker installed:

- **Windows/Mac**: Download [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- **Linux**: Run `sudo apt install docker.io`

Verify installation:
```bash
docker --version
```

---

### Step 2: Load the Docker Image

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
==================================================
Solar Flare API Client Example (Random CSV Row)
==================================================

1. Checking API health...
   Status: healthy
   Model loaded: True
   Expected features: ['flare', 'peak.c/s', 'total.counts', ...]

2. Predicting using random CSV row...
   Prediction class: 2
   Prediction label: Long (>=400s)
   Features used: {'flare': 2021213.0, 'peak.c/s': 136.0, ...}

==================================================
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

| Problem | Solution |
|---------|----------|
| "Cannot connect to API" | Make sure Docker container is running |
| "Docker not found" | Install Docker Desktop |
| "Port 8000 in use" | Use `docker run -p 8001:8000 solar-flare-api` and update `API_URL` in client |
| CSV file not found | Put the CSV in the same folder as `client_example.py` |

---

## 📝 Using Your Own Data

To predict with your own data in Python:

```python
import requests
import pandas as pd

# Load your CSV (must have same columns as training data)
df = pd.read_csv("your_data.csv")

# Get any row
row = df.iloc[0].to_dict()

# Make prediction
response = requests.post("http://localhost:8000/predict", json={"row": row})
print(response.json())
```

The API automatically extracts only the 7 features it needs from your full row.

---

## 🎯 That's It!

You've successfully:
1. ✅ Loaded the Docker image
2. ✅ Started the API server
3. ✅ Made predictions using Python

For more details, visit http://localhost:8000/docs when the API is running.
