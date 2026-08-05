# ScreenRedactor

**ScreenRedactor** is a real-time screen censorship application designed for streaming software (e.g., OBS Studio) and video conferencing platforms (e.g., Zoom, Discord, Teams). The program captures screen content, detects sensitive information using optical character recognition (OCR), and routes the redacted video stream directly to a virtual camera.

---
## Features

* **Real-Time Censorship:** Automatically detects and obscures sensitive data (e.g., passwords, emails, API keys) on the fly.
* **Virtual Camera Output:** Integrates seamlessly as a video source in OBS, Discord, Teams, or Zoom.
* **One-Click Start:** Easy launching via the included batch file (`start_screen_redactor.bat`).
* **Standalone Executable:** Can be compiled directly into an `.exe` file using the built-in build script.

---

## How It Works

1. **Capture:** The application captures the selected screen or window in real time.
2. **Analysis:** Text recognition and image processing locally analyze the content for confidential data.
3. **Redaction:** Detected sensitive details are automatically covered with blur or blackout blocks.
4. **Output:** The processed video stream is output as a virtual camera feed for other applications.

---

## Installation & Usage

### Prerequisites
* Windows 10 / 11
* Python 3.10 or newer
* Virtual camera driver (e.g., **OBS Virtual Camera**)

### Running the Application
1. Clone the repository:
   cmd
   git clone [https://github.com/KecksKruemmel/ScreenRedactor.git](https://github.com/KecksKruemmel/ScreenRedactor.git)
   cd ScreenRedactor

Launch via the batch file or terminal:
Double-click start_screen_redactor.bat or run:
python realtime_screen_redactor_virtualcam.py

Building the Executable (.exe)
To create a standalone Windows application without requiring a local Python setup:

Run build_screen_redactor_exe.bat.

The compiled application will be generated in the local dist/ directory.

Tech Stack
Language: Python

Image Processing & "AI"": OpenCV, PyTorch

Packaging: PyInstaller