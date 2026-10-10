@echo off
title Deepfake Detection v2.0 - Frontend Web UI
cd /d "C:\Users\Administrator\Documents\Deepfake-Detection\frontend"
echo ===================================================
echo Starting Forensic Web UI on port 8080...
echo Phone / Network Access: http://192.168.1.5:8080 or http://localhost:8080
echo ===================================================
call npm.cmd run dev -- --host
pause
