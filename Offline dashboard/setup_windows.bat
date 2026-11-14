@echo off
setlocal
cd /d "%~dp0"
echo AquaVision - Windows setup / repair
echo Internet is required for dependency installation.
if exist .venv\Scripts\python.exe goto install
py -3.12 -m venv .venv
if not errorlevel 1 goto install
py -3.11 -m venv .venv
if not errorlevel 1 goto install
py -m venv .venv
if errorlevel 1 goto failed
:install
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 goto failed
.venv\Scripts\python.exe -c "import torch, torchvision" >nul 2>&1
if not errorlevel 1 goto packages
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto failed
:packages
.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 goto failed
.venv\Scripts\python.exe check_dependencies.py
if errorlevel 1 goto failed
echo Setup complete. Double-click run.bat.
pause
exit /b 0
:failed
echo Setup did not finish. Read the error above. Use Python 3.11 or 3.12 64-bit.
pause
exit /b 1
