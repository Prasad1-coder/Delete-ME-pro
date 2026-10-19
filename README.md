# DeleteMe Pro — Real-Time Invisibility System

**DeleteMe Pro** is a Python-based computer vision project that creates real-time invisibility and ghost effects using a webcam, background capture, and hand gesture recognition.

The project combines computer vision and image processing to create interactive visual effects through a desktop application.

## Features

* **Real-Time Invisibility Effect:** Makes the detected person appear invisible by replacing the person region with the captured background.
* **Ghost Effect:** Applies a semi-transparent visual effect.
* **Hand Gesture Recognition:** Uses hand gestures to control visual modes.
* **Background Capture:** Captures the environment to generate the invisibility effect.
* **Background Recapture:** Allows the background to be captured again when the environment changes.
* **Computer Vision Processing:** Uses image processing and person segmentation techniques.
* **Desktop Interface:** Provides an interactive graphical user interface.
* **Live Webcam Processing:** Processes camera frames in real time.

## Technologies Used

* Python
* OpenCV
* MediaPipe
* NumPy
* PySide6

## Project Structure

```text
DeleteMe-Pro/
├── main.py
├── deleteme_v7.py
├── invisibility.py
├── hand_tracking.py
├── recordings/
├── screenshots/
└── README.md
```

*Note: The project structure may vary depending on the files included in the repository.*

## How It Works

1. The application accesses the webcam.
2. It captures the background without the person in the frame.
3. Computer vision techniques detect the person and process the camera feed.
4. Hand gestures allow the user to switch between visual modes.
5. The application combines the processed person mask with the captured background to create invisibility and ghost effects.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Prasad1-coder/Delete-ME-pro.git
```

### 2. Open the Project Directory

```bash
cd Delete-ME-pro
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

**Windows PowerShell:**

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

Install the required packages:

```bash
pip install opencv-python mediapipe numpy PySide6
```

### 6. Run the Application

Check the project files and run the appropriate entry-point script:

```bash
python main.py
```

If `main.py` is not the application entry point, run the script that initializes the DeleteMe Pro graphical interface.

## Controls

The following controls are used in the current interface:

| Key | Action               |
| --- | -------------------- |
| `1` | Normal Mode          |
| `2` | Invisibility Mode    |
| `3` | Ghost Mode           |
| `B` | Capture Background   |
| `V` | Recapture Background |
| `7` | HD Resolution        |
| `8` | Full HD Resolution   |
| `S` | Take Screenshot      |
| `Q` | Quit Application     |

*Controls may vary depending on the version being used.*

## Screenshots

Project screenshots are available in the `screenshots/` directory.

Add screenshots here to demonstrate the application's interface and visual effects.

## Future Improvements

* Improve performance under changing lighting conditions.
* Enhance person segmentation and edge refinement.
* Improve background analysis and recapture.
* Optimize real-time processing performance.
* Add more gesture controls and visual effects.

## Learning Outcomes

This project provides practical experience with:

* Python programming
* Computer vision and image processing
* Real-time webcam processing
* Hand gesture recognition
* Person segmentation
* Desktop GUI development
* Integrating multiple Python libraries

## Author

**Prasad Deshmane**

B.Sc. Computer Science Student

GitHub: [Prasad1-coder](https://github.com/Prasad1-coder)

## Disclaimer

DeleteMe Pro is an experimental computer vision project developed for educational and demonstration purposes. Its visual effects depend on camera quality, lighting conditions, background stability, and processing performance.
