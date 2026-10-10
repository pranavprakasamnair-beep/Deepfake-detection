@echo off
title PCE Forensic Deepfake Detection Engine - Unified Launcher
cd /d "C:\Users\Administrator\Documents\Deepfake-Detection"

echo ======================================================================
echo   PILLAI COLLEGE OF ENGINEERING (PCE AUTONOMOUS)
echo   Department of Computer Engineering - B.Tech Final Year Capstone
echo   Multi-Branch Deepfake Detection & Forensic Verification System
echo ======================================================================
echo.
echo Launching Backend Forensic Engine (Port 8000)...
start "PCE Forensic Backend" cmd /c "RUN_BACKEND.bat"

timeout /t 3 /nobreak >nul

echo Launching Frontend Interface (Port 8080)...
start "PCE Forensic Frontend" cmd /c "RUN_FRONTEND.bat"

echo.
echo ======================================================================
echo   Deployment successful!
echo   Local Web Access:   http://localhost:8080
echo   Mobile/LAN Access:  http://192.168.1.5:8080
echo   Backend API Docs:   http://localhost:8000/docs
echo ======================================================================
echo.
