@echo off
setlocal
cd /d "%~dp0"
echo [Novel Analyzer] Starting...

set "VENV_DIR=%~dp0venv"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"
set "PIP=%VENV_DIR%\Scripts\pip.exe"

if not exist "%VENV_DIR%" (
  echo [INFO] Creating venv...
  python -m venv "%VENV_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to create venv.
    pause
    exit /b 1
  )
)

if not exist "%PIP%" (
  echo [INFO] Installing pip in venv...
  "%PYTHON%" -m ensurepip --upgrade
)

"%PYTHON%" -m pip install -q --disable-pip-version-check -r requirements.txt 2>nul
if errorlevel 1 (
  echo [INFO] Installing dependencies...
  "%PYTHON%" -m pip install --disable-pip-version-check -r requirements.txt
  if errorlevel 1 (
    echo [ERROR] Dependency install failed.
    pause
    exit /b 1
  )
)

"%PYTHON%" backend.py
pause
