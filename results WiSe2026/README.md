# Vision Assistant - Client-Server Architecture

This application has been restructured into a client-server architecture to separate the user interface from the heavy AI model processing.

## Architecture Overview

```
┌─────────────────┐     HTTP API     ┌─────────────────┐
│     CLIENT      │ ◄──────────────► │     SERVER      │
│                 │                  │                 │
│ • Streamlit UI  │                  │ • AI Models     │
│ • Camera        │                  │ • Processing    │
│ • Voice UI      │                  │ • TTS           │
│ • Display       │                  │ • FastAPI       │
└─────────────────┘                  └─────────────────┘
```

## Components

### Server (`/server`)
- **FastAPI backend** that handles all AI model processing
- **Model loading**: YOLO, face recognition, scene description
- **Frame processing**: Receives frames, processes them, returns results
- **Text-to-speech**: Handles audio output
- **API endpoints** for communication with client

### Client (`/client`) 
- **Streamlit web interface** for user interaction
- **Camera capture** and video display
- **Voice command processing** for hands-free operation
- **HTTP communication** with server for frame processing
- **Real-time video feed** with processed results

## Quick Setup (Automated)

### Windows
```cmd
# Run the automated setup script
setup_scripts\setup_windows.bat
```

### Ubuntu/Linux
```bash
# Make script executable and run
chmod +x setup_scripts/setup_ubuntu.sh
./setup_scripts/setup_ubuntu.sh
```

### macOS
```bash
# Make script executable and run
chmod +x setup_scripts/setup_macos.sh
./setup_scripts/setup_macos.sh
```

## Environment Setup (Manual)

### Windows Setup

#### Prerequisites
```bash
# Install Python 3.8+ from https://python.org
# Install Git from https://git-scm.com

# Open Command Prompt or PowerShell as Administrator
```

#### Create Virtual Environments
```bash
# Create server environment
cd server
python -m venv venv_server
venv_server\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# Create client environment (in new terminal)
cd ..\client
python -m venv venv_client
venv_client\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Install System Dependencies (Windows)
```bash
# Install Visual Studio Build Tools for dlib
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
# Install with C++ build tools

# Alternative: Use conda for easier dlib installation
conda create -n vision_assistant python=3.9
conda activate vision_assistant
conda install -c conda-forge dlib
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

### Ubuntu/Debian Setup

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv git
sudo apt install -y build-essential cmake pkg-config
sudo apt install -y libopenblas-dev liblapack-dev
sudo apt install -y libx11-dev libgtk-3-dev
sudo apt install -y libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt install -y libxvidcore-dev libx264-dev
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev
sudo apt install -y libatlas-base-dev gfortran
sudo apt install -y portaudio19-dev python3-pyaudio
sudo apt install -y espeak espeak-data
```

#### Create Virtual Environments
```bash
# Create server environment
cd server
python3 -m venv venv_server
source venv_server/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create client environment (in new terminal)
cd ../client
python3 -m venv venv_client
source venv_client/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### macOS Setup

#### Prerequisites
```bash
# Install Homebrew (if not installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install system dependencies
brew install python3
brew install cmake
brew install pkg-config
brew install openblas
brew install lapack
brew install portaudio
brew install espeak
brew install ffmpeg
```

#### Create Virtual Environments
```bash
# Create server environment
cd server
python3 -m venv venv_server
source venv_server/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Create client environment (in new terminal)
cd ../client
python3 -m venv venv_client
source venv_client/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Quick Start

### 1. Start the Server
```bash
# Windows
cd server
venv_server\Scripts\activate
python start_server.py

# Ubuntu/macOS
cd server
source venv_server/bin/activate
python start_server.py
```

### 2. Start the Client
```bash
# Windows (in new terminal)
cd client
venv_client\Scripts\activate
python start_client.py

# Ubuntu/macOS (in new terminal)
cd client
source venv_client/bin/activate
python start_client.py
```

### 3. Access the Application
- Open your browser to `http://localhost:8501`
- Server health check: `http://localhost:8000/health`

## Required Files

Make sure you have these files in the project root:
- `commands.json` - Voice commands configuration
- `models/` directory with AI model files
- `known_faces/` directory with face images (optional)

## Troubleshooting

### Common Issues

#### dlib Installation Issues (Windows)
```bash
# Use conda instead of pip for dlib
conda install -c conda-forge dlib
```

#### Audio Issues (Ubuntu)
```bash
# Install additional audio dependencies
sudo apt install -y pulseaudio pulseaudio-utils
sudo apt install -y libasound2-dev
```

#### Camera Access Issues (macOS)
```bash
# Grant camera permissions in System Preferences > Security & Privacy > Camera
# Add Terminal/VS Code to allowed applications
```

#### Model Download Issues
```bash
# Manual model download
mkdir -p models/speech/model
cd models/speech/model
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model-small-en-us-0.15
```

## Features

- **Object Detection**: Real-time object identification
- **Face Recognition**: Recognize known people
- **Scene Description**: AI-generated scene descriptions
- **Voice Commands**: Hands-free operation
- **Text-to-Speech**: Audio feedback
- **Real-time Processing**: Live video feed with AI analysis

## Benefits of Client-Server Architecture

1. **Scalability**: Server can handle multiple clients
2. **Resource Management**: Heavy AI models run on server only
3. **Flexibility**: Client and server can run on different machines
4. **Maintainability**: Clear separation of concerns
5. **Performance**: Better resource utilization

## Original Functionality Preserved

All original functionality from `vision_app.py` has been preserved:
- Same AI models and processing logic
- Same user interface and controls
- Same voice commands and TTS
- Same real-time performance
- Same features and capabilities

The code has simply been reorganized into a more scalable architecture without changing any core functionality. 