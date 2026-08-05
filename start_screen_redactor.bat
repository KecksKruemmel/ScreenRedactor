@echo off
cd /d "%~dp0"

echo [1/4] Entferne evtl. vorhandenes onnxruntime (CPU-Version) ...
python -m pip uninstall -y onnxruntime

echo.
echo [2/4] Installiere/aktualisiere benoetigte Pakete ...
python -m pip install --upgrade customtkinter pyvirtualcam mss opencv-python numpy rapidocr onnxruntime-directml
if errorlevel 1 (
    echo.
    echo FEHLER: Paketinstallation fehlgeschlagen. Siehe Meldung oben.
    echo Falls dort "invalid distribution ~nnxruntime" steht: alle python.exe
    echo im Taskmanager beenden, dann im Ordner
    echo   %%LOCALAPPDATA%%\Programs\Python\Python313\Lib\site-packages
    echo jeden Ordner loeschen, der mit ~nnxruntime beginnt. Danach dieses
    echo Skript erneut starten.
    pause
    exit /b 1
)

echo.
echo [3/4] Pruefe DirectML Verfuegbarkeit ...
python -c "import onnxruntime as ort; p=ort.get_available_providers(); print('Provider:', p); print('DirectML verfuegbar' if 'DmlExecutionProvider' in p else 'DirectML NICHT verfuegbar, es wird CPU verwendet')"

echo.
echo [4/4] Starte ScreenRedactor ...
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
python realtime_screen_redactor_virtualcam.py
if errorlevel 1 (
    echo.
    echo Das Programm wurde mit einem Fehler beendet, siehe Meldung oben.
)

echo.
echo ScreenRedactor stopped.
pause