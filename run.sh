#!/bin/bash

# Alzheimer's Detection System - Startup Script
# This script sets up and runs the application

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║      ALZHEIMER'S DISEASE DETECTION SYSTEM                ║"
echo "║      AI-Powered Brain Scan Analysis                      ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null
then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if required packages are installed
echo "Checking dependencies..."

REQUIRED_PACKAGES=("flask" "tensorflow" "numpy" "cv2" "PIL")
MISSING_PACKAGES=()

for package in "${REQUIRED_PACKAGES[@]}"
do
    if ! python3 -c "import $package" 2>/dev/null
    then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]
then
    echo "❌ Missing packages: ${MISSING_PACKAGES[*]}"
    echo ""
    echo "Installing dependencies..."
    pip install -r requirements.txt
    
    if [ $? -ne 0 ]
    then
        echo "❌ Failed to install dependencies. Please install manually:"
        echo "   pip install -r requirements.txt"
        exit 1
    fi
    echo "✓ Dependencies installed successfully"
else
    echo "✓ All dependencies are installed"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Starting Alzheimer's Detection System..."
echo "════════════════════════════════════════════════════════════"
echo ""

# Run the application
python3 app.py