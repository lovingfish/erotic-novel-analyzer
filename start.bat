@echo off
setlocal
cd /d "%~dp0"
echo ========================================
echo Novel Analyzer - Launcher
echo ========================================

set "PYTHON=%~dp0venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

echo [1/2] Installing dependencies...
"%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
if errorlevel 1 (
  echo Dependency install failed.
  pause
  exit /b 1
)

echo [2/2] Starting server...
echo.
echo Open the Local URL printed by the server.
echo Press Ctrl+C to stop.
echo ========================================

"%PYTHON%" backend.py

pause
