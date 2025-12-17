@echo off
echo ==================================================
echo 🌞 Solar Flare Prediction App - Launcher
echo ==================================================
echo.

echo 1. Installing dependencies (first run only)...
pip install -r requirements.txt
echo.

echo 2. Starting Server...
echo --------------------------------------------------
echo The app will be available at: http://localhost:8000
echo Press Ctrl+C to stop the server.
echo --------------------------------------------------
python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload
pause
