# Real-Time Object Detection with Audio Alerts

A real-time object detection system that uses YOLOv8 to identify objects through your webcam and provides audio alerts via text-to-speech (TTS) when objects are detected.

## Features

-  **Real-time object detection** using YOLOv8 neural network
-  **Live webcam feed** with object detection overlay
-  **Audio alerts** with customizable TTS announcements
-  **Smart cooldown system** to prevent alert spam
-  **Performance optimized** with frame skipping for smooth operation
-  **Automatic cleanup** of temporary audio files
-  **Configurable detection confidence** threshold

## Prerequisites

Before running this project, ensure you have:

- **Python 3.8+** installed
- **Webcam** connected and accessible
- **TTS API server** running on `localhost:8000` (see [TTS Setup](#tts-setup))
- **Audio output device** for playing alerts

## Installation

1. **Clone or download** this repository

2. **Install required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model** (happens automatically on first run):
   - The script will download `yolov8n.pt` model file automatically
   - Alternatively, you can pre-download it from [Ultralytics](https://github.com/ultralytics/ultralytics)

## TTS Setup

This project requires a TTS (Text-to-Speech) API running on `localhost:8000`. The API should:

- Accept POST requests to `/speak` endpoint
- Accept `text` parameter in request body
- Return WAV audio data in response

### Example TTS API Setup

You can use various TTS solutions. Here's a simple example using a Python TTS server:

## Usage

1. **Start your TTS API server** on `localhost:8000`

2. **Run the object detection system**:
   ```bash
   python object-detection.py
   ```

3. **Position your webcam** to capture the area you want to monitor

4. **Listen for audio alerts** when objects are detected

5. **Press 'q'** in the video window to quit the application

## Configuration

### Adjustable Parameters

You can modify these variables in `object-detection.py`:

- **`frame_skip = 10`**: Process every Nth frame (higher = better performance, lower accuracy)
- **`cooldown_seconds = 5`**: Minimum time between alerts for the same object type
- **`conf=0.5`**: Detection confidence threshold (0.0-1.0)

### Detection Classes

The system can detect 80+ object classes including:
- People
- Vehicles (car, truck, bus, motorcycle, bicycle)
- Animals (dog, cat, bird, horse, etc.)
- Common objects (bottle, chair, laptop, phone, etc.)

## Project Structure

```
Object_detection/
├── object-detection.py    # Main detection script
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── yolov8n.pt           # YOLOv8 model (downloaded automatically)
```

## How It Works

1. **Webcam Capture**: Uses OpenCV to capture live video feed
2. **Object Detection**: YOLOv8 model processes frames to identify objects
3. **Alert Generation**: Detected objects trigger TTS API calls
4. **Audio Playback**: Generated audio is played through speakers
5. **Cooldown Management**: Prevents duplicate alerts for recently detected objects
6. **Cleanup**: Temporary audio files are automatically removed

## Performance Optimization

- **Frame Skipping**: Processes every 10th frame to maintain smooth performance
- **Confidence Filtering**: Only processes detections above 50% confidence
- **Efficient Model**: Uses YOLOv8n (nano) model for fast inference
- **Memory Management**: Automatic cleanup of temporary audio files

## Troubleshooting

### Common Issues

**Webcam not working:**
- Check if webcam is connected and not used by other applications
- Try changing `cv2.VideoCapture(0)` to `cv2.VideoCapture(1)` for different camera

**TTS API connection failed:**
- Ensure TTS server is running on `localhost:8000`
- Check firewall settings
- Verify API endpoint accepts POST requests to `/speak`

**Audio playback issues:**
- Check audio drivers and output devices
- Ensure `simpleaudio` can access audio system
- Try running with administrator privileges on Windows

**Performance issues:**
- Increase `frame_skip` value for better performance
- Lower detection confidence threshold
- Use a more powerful GPU if available

**File deletion errors:**
- Windows may lock audio files briefly - the script includes retry logic
- Ensure sufficient disk permissions

## Dependencies

- **ultralytics**: YOLOv8 object detection model
- **opencv-python**: Computer vision and webcam handling
- **requests**: HTTP requests to TTS API
- **simpleaudio**: Cross-platform audio playback

## Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the YOLOv8 model
- [OpenCV](https://opencv.org/) for computer vision capabilities 