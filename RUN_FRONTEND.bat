@echo off
title Deepfake Detection v2.0 - Frontend Web UI
cd /d "C:\Users\Administrator\Documents\Deepfake-Detection\frontend"
echo ===================================================
echo Starting DeepGuard Web UI on port 8080...
echo Phone Access: http://192.168.1.7:8080
echo ===================================================
call npm.cmd run dev -- --host
pause
