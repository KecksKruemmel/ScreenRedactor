@echo off
cd /d "%~dp0"
python -m pip install --quiet --upgrade pyvirtualcam mss opencv-python numpy rapidocr onnxruntime
set OMP_NUM_THREADS=2
set MKL_NUM_THREADS=2
set OPENBLAS_NUM_THREADS=2
python realtime_screen_redactor_virtualcam.py --monitor 1 --fps 30 --mode black --ttl 12 --ocr-every 4 --ocr-max-width 1600 --min-confidence 0.3 --model-tier small --lang en
echo.
echo ScreenRedactor stopped.
pause