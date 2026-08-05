@echo off
setlocal

cd /d "%~dp0"

echo.
echo Building ScreenRedactor.exe
echo This is fast: RapidOCR uses small ONNXRuntime models.
echo.

where python >nul 2>nul
if errorlevel 1 (
    echo Python was not found in PATH.
    echo Open PowerShell and check that "python --version" works first.
    pause
    exit /b 1
)

python -m pip install --upgrade pyinstaller
if errorlevel 1 (
    echo Failed to install or update PyInstaller.
    pause
    exit /b 1
)

python -m PyInstaller ^
  --noconfirm ^
  --clean ^
  --onedir ^
  --name ScreenRedactor ^
  --collect-all rapidocr ^
  --collect-all onnxruntime ^
  --collect-all customtkinter ^
  --collect-all pyvirtualcam ^
  --collect-submodules mss ^
  --collect-submodules cv2 ^
  realtime_screen_redactor_virtualcam.py

if errorlevel 1 (
    echo.
    echo Build failed.
    pause
    exit /b 1
)

echo.
echo Done.
echo EXE:
echo %CD%\dist\ScreenRedactor\ScreenRedactor.exe
echo.
pause