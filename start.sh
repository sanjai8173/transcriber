#!/bin/bash

clear
echo
echo " ╔══════════════════════════════════════════╗"
echo " ║   VOXSTREAM  —  Starting Server         ║"
echo " ╚══════════════════════════════════════════╝"
echo

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo " [INFO] Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo " [INFO] Activating virtual environment..."
source .venv/bin/activate

# Install dependencies if uvicorn is missing
if ! command -v uvicorn >/dev/null 2>&1; then
    echo " [INFO] Installing dependencies..."
    pip install -r requirements.txt
fi

echo " [INFO] Starting server on http://localhost:8000"
echo " [INFO] Press Ctrl+C to stop"
echo

uvicorn main:app --host 0.0.0.0 --port 8000 --reload
