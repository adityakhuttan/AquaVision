@echo off
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo Run setup_windows.bat first, then run.bat.
  pause
  exit /b 1
)
.venv\Scripts\python.exe check_dependencies.py
if errorlevel 1 (
  pause
  exit /b 1
)
.venv\Scripts\python.exe desktop_app.py
if errorlevel 1 pause
