# 🎧 Voice-Only Vision Assistant for Blind People

A Streamlit-based user interface for an audio-driven vision assistance system designed to help visually impaired users navigate their environment through voice announcements.

## Features

### Core Functionality
- **Object Detection**: Real-time identification and audio description of objects in view
- **Face Recognition**: Identification of known individuals with voice announcements
- **Scene Description**: Comprehensive audio descriptions of the current environment
- **Voice Command Control**: Hands-free operation through voice commands

### System Monitoring
- **Battery Status**: Real-time battery level monitoring with low-battery warnings
- **CPU Usage**: System performance monitoring
- **Temperature Monitoring**: Hardware temperature tracking
- **Server Health**: Connection status to backend vision processing server

### User Interface
- **Audio Messages Log**: Real-time display of all voice announcements
- **Control Panel**: Easy-to-use buttons for mode switching
- **Status Indicators**: Clear visual feedback for current operating mode
- **Responsive Design**: Wide layout optimized for accessibility

## Target Users

This application is specifically designed for:
- Visually impaired individuals
- Blind users requiring environmental awareness
- Caregivers and assistants
- Accessibility researchers and developers

##  Quick Start

### Prerequisites

- Python 3.7 or higher
- Webcam or camera device (for actual vision processing)
- Microphone (for voice commands)
- Speakers or headphones (for audio feedback)

### Installation

1. **Clone or download this repository**
   ```bash
   git clone <repository-url>
   cd UI
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   python run_streamlit.py
   ```
   
   Or alternatively:
   ```bash
   streamlit run run_streamlit.py
   ```

4. **Access the interface**
   - Open your web browser
   - Navigate to `http://localhost:8501`
   - The application will load automatically

##  Usage

### Basic Operation

1. **Start Voice Commands**: Click the "🎤 Start Voice Commands" button to enable voice control
2. **Select Detection Mode**: Choose from:
   - **Object Detection**: Identifies and announces objects
   - **Face Recognition**: Recognizes known faces
   - **Scene Description**: Provides comprehensive environmental descriptions
3. **Monitor Audio Log**: All announcements appear in the messages panel
4. **System Status**: Keep an eye on battery, CPU, and temperature indicators

### Voice Commands

The system supports voice activation for:
- Starting/stopping different detection modes
- Requesting immediate scene descriptions
- System status inquiries
- Emergency stop commands

### Controls

| Button            | Function                       |
|-------------------|--------------------------------|
|  Voice Commands   | Toggle voice command listening |
|  Object Detection | Start identifying objects      |
|  Face Recognition | Begin face recognition mode    |
|  Scene Description| Generate scene descriptions    |
|  Stop Detection   | Halt all detection activities  |
|  Clear Messages   | Clear the audio messages log   |

##  Configuration

### System Requirements

- **Minimum RAM**: 4GB (8GB recommended)
- **CPU**: Multi-core processor recommended for real-time processing
- **Storage**: 2GB free space for models and cache
- **Network**: Internet connection for initial model downloads

### Performance Optimization

- **Battery Monitoring**: Automatic warnings below 12% battery
- **CPU Throttling**: Automatic performance adjustment based on system load
- **Temperature Management**: Thermal monitoring to prevent overheating

##  Technical Details

### Architecture

- **Frontend**: Streamlit web interface
- **Backend**: Separate vision processing server (not included in this UI demo)
- **Communication**: RESTful API or WebSocket connections
- **Audio**: Text-to-speech synthesis for announcements

### Current Status

 **Note**: This is currently a **static UI demonstration**. The interface shows simulated data and responses. For full functionality, a backend vision processing server needs to be implemented and connected.

### Data Flow

1. Camera captures video feed
2. Backend processes images using AI models
3. Results are converted to audio announcements
4. UI displays status and logs all activities
5. Voice commands control system behavior

## Development

### Project Structure

```
UI/
├── run_streamlit.py     # Main Streamlit application
├── requirements.txt     # Python dependencies
└── README.md           # This documentation
```

### Adding Features

To extend functionality:

1. **New Detection Modes**: Add buttons and handlers in the control panel
2. **Custom Voice Commands**: Extend the voice command processor
3. **Enhanced Logging**: Modify the messages system for better tracking
4. **Settings Panel**: Add configuration options for advanced users

### Backend Integration

To connect with a real vision processing backend:

1. Replace static data with API calls
2. Implement WebSocket connections for real-time updates
3. Add error handling for network connectivity
4. Include authentication for secure access

## Accessibility Features

- **High Contrast UI**: Clear visual indicators for low vision users
- **Large Buttons**: Easy-to-target interface elements
- **Audio-First Design**: Primary interaction through voice
- **Keyboard Navigation**: Full keyboard accessibility support
- **Screen Reader Compatible**: Semantic HTML for assistive technologies

## Troubleshooting

### Common Issues

**Application won't start**
- Check Python version (3.7+ required)
- Verify all dependencies are installed
- Ensure port 8501 is available

**No audio output**
- Check speaker/headphone connections
- Verify system audio settings
- Test with other audio applications

**High CPU usage**
- Reduce detection frequency
- Close unnecessary applications
- Check for background processes

**Battery draining quickly**
- Lower screen brightness
- Reduce processing intensity
- Enable power saving mode
