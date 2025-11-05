@echo off
REM Alzheimer's Detection System - Windows Startup Script

echo ================================================================
echo.
echo       ALZHEIMER'S DISEASE DETECTION SYSTEM
echo       AI-Powered Brain Scan Analysis
echo.
echo ================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask" >nul 2>&1
if %errorlevel% neq 0 (
    echo [MISSING] Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to install dependencies
        echo Please run manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies are installed
)

echo.
echo ================================================================
echo Starting Alzheimer's Detection System...
echo ================================================================
echo.
echo Once started, open your browser and go to:
echo http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run the application
python app.py

pause