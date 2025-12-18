@echo off
SETLOCAL

echo ==================================================
echo 🌞 Solar Flare Prediction App - Launcher
echo ==================================================
echo.

REM --------------------------------------------------
REM Configuration
REM --------------------------------------------------
set VENV_DIR=.venv

REM --------------------------------------------------
REM Step 1: Create venv if it doesn't exist
REM --------------------------------------------------
if not exist %VENV_DIR% (
    echo 🔧 Creating virtual environment...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
) else (
    echo ✅ Virtual environment already exists
)

echo.

REM --------------------------------------------------
REM Step 2: Activate venv
REM --------------------------------------------------
echo 🔄 Activating virtual environment...
call %VENV_DIR%\Scripts\activate.bat
if errorlevel 1 (
    echo ❌ Failed to activate virtual environment
    pause
    exit /b 1
)

echo.

REM --------------------------------------------------
REM Step 3: Upgrade pip (recommended)
REM --------------------------------------------------
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

echo.

REM --------------------------------------------------
REM Step 4: Install dependencies
REM --------------------------------------------------
echo 📦 Installing dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo ❌ Dependency installation failed
    pause
    exit /b 1
)

echo.

REM --------------------------------------------------
REM Step 5: Start FastAPI server
REM --------------------------------------------------
echo 🚀 Starting FastAPI server...
echo --------------------------------------------------
echo App running at: http://localhost:8000
echo Docs available at: http://localhost:8000/docs
echo Press Ctrl+C to stop the server
echo --------------------------------------------------

python -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

pause
ENDLOCAL
