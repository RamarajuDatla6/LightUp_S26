# Vision Assistant Server

This is the server component of the Vision Assistant application that handles all the heavy computation including:

- AI model loading and inference (YOLO, face recognition, scene description)
- Frame processing
- Text-to-speech functionality

## Environment Setup

### Windows Setup

#### Prerequisites
```bash
# Install Python 3.8+ from https://python.org
# Install Visual Studio Build Tools for dlib compilation
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
```

#### Create Virtual Environment
```bash
# Create and activate virtual environment
python -m venv venv_server
venv_server\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Alternative: Using Conda (Recommended for Windows)
```bash
# Create conda environment
conda create -n vision_server python=3.9
conda activate vision_server
conda install -c conda-forge dlib
pip install -r requirements.txt
```

### Ubuntu/Debian Setup

#### Prerequisites
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3 python3-pip python3-venv
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

#### Create Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv venv_server
source venv_server/bin/activate
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

#### Create Virtual Environment
```bash
# Create and activate virtual environment
python3 -m venv venv_server
source venv_server/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup

1. Install dependencies (see Environment Setup above)

2. Make sure you have the required model files in the `models/` directory:
- `models/deploy.prototxt`
- `models/res10_300x300_ssd_iter_140000.caffemodel`
- `models/shape_predictor_68_face_landmarks.dat`
- `models/dlib_face_recognition_resnet_model_v1.dat`
- `models/speech/model/vosk-model-small-en-us-0.15/` (for speech recognition)

3. Create a `known_faces/` directory and add face images for recognition (optional)

4. Create a `commands.json` file with voice commands (copy from main directory)

## Running the Server

### Windows
```bash
# Activate environment
venv_server\Scripts\activate
# or
conda activate vision_server

# Start server
python start_server.py
```

### Ubuntu/macOS
```bash
# Activate environment
source venv_server/bin/activate

# Start server
python start_server.py
```

### Alternative Methods
```bash
# Direct execution
python server.py

# Using uvicorn directly
uvicorn server:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://localhost:8000`

## API Endpoints

- `POST /process_frame` - Process a frame with AI models
- `POST /speak` - Text-to-speech
- `GET /health` - Health check

## Health Check

Visit `http://localhost:8000/health` to check if the server is running and all models are loaded properly.

## Troubleshooting

### Common Issues

#### dlib Installation Issues (Windows)
```bash
# Use conda instead of pip
conda install -c conda-forge dlib
```

#### Audio Issues (Ubuntu)
```bash
# Install additional audio dependencies
sudo apt install -y pulseaudio pulseaudio-utils
sudo apt install -y libasound2-dev
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

#### CUDA Issues
```bash
# If you have CUDA GPU, install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
``` 