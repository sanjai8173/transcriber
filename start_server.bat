@echo off
title VoxStream — Transcription Server
echo.
echo  ╔══════════════════════════════════════════╗
echo  ║   VOXSTREAM  —  Starting Server         ║
echo  ╚══════════════════════════════════════════╝
echo.

:: Create venv if it doesn't exist
if not exist ".venv\" (
    echo  [INFO] Creating virtual environment...
    python -m venv .venv
)

:: Activate venv
echo  [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Install dependencies if uvicorn is missing in local venv
if not exist ".venv\Scripts\uvicorn.exe" (
    echo  [INFO] Installing dependencies...
    pip install -r requirements.txt
)

echo  [INFO] Starting server on http://localhost:8000
echo  [INFO] Press Ctrl+C to stop
echo.
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
pause
