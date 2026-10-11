@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   Audio Upscaler HD - full auto setup
echo   (Python deps + ffmpeg + GPU runtime)
echo ============================================
echo.

where py >nul 2>nul && set PY=py || set PY=python
where %PY% >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.11+ from python.org first.
    pause & exit /b 1
)

echo [1/4] Creating virtual environment ...
%PY% -m venv .venv
if errorlevel 1 ( echo venv creation failed. & pause & exit /b 1 )

call .venv\Scripts\activate.bat

echo [2/4] Upgrading pip ...
python -m pip install --upgrade pip --quiet

echo [3/4] Installing Python dependencies (numpy, scipy, onnxruntime-directml, GUI, ...) ...
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo WARNING: some Python packages failed to install ^(see above^).
    echo The DSP engine may still work; re-run this script after fixing.
) else (
    echo       Python dependencies OK.
)
echo.

echo [4/4] Installing ffmpeg (winget or direct download) ...
python scripts\install_ffmpeg.py
echo.

echo === Setup finished ===
echo   GUI:  run_gui.bat
echo   CLI:  .venv\Scripts\python main.py cli --help
echo   Test: .venv\Scripts\python -m pytest tests -q
echo.
pause
