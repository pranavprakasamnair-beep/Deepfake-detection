@echo off
title Deepfake Detection v2.0 - Backend Server
cd /d "C:\Users\Administrator\Documents\Deepfake-Detection\backend"
echo ===================================================
echo Starting DeepGuard Forensic Engine on port 8000...
echo ===================================================
".\.venv\Scripts\python.exe" -m uvicorn main:app --host 0.0.0.0 --port 8000
pause
