#!/bin/bash

echo "========================================"
echo "Vision Assistant - macOS Setup Script"
echo "========================================"
echo

# Check Python installation
echo "Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python3 is not installed"
    echo "Please install Python3: brew install python3"
    exit 1
fi

echo "Python found: $(python3 --version)"
echo

# Check Homebrew installation
echo "Checking Homebrew installation..."
if ! command -v brew &> /dev/null; then
    echo "Installing Homebrew..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    
    # Add Homebrew to PATH for Apple Silicon Macs
    if [[ $(uname -m) == 'arm64' ]]; then
        echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

echo "Homebrew found: $(brew --version)"
echo

# Install system dependencies
echo "Installing system dependencies..."
brew install python3
brew install cmake
brew install pkg-config
brew install openblas
brew install lapack
brew install portaudio
brew install espeak
brew install ffmpeg

echo

# Create server environment
echo "Creating server environment..."
cd server
python3 -m venv venv_server
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create server virtual environment"
    exit 1
fi

echo "Activating server environment..."
source venv_server/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate server environment"
    exit 1
fi

echo "Installing server dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "WARNING: Some server dependencies failed to install"
    echo "You may need to install additional system packages"
fi

echo

# Create client environment
echo "Creating client environment..."
cd ../client
python3 -m venv venv_client
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create client virtual environment"
    exit 1
fi

echo "Activating client environment..."
source venv_client/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate client environment"
    exit 1
fi

echo "Installing client dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install client dependencies"
    exit 1
fi

echo
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo
echo "To start the server:"
echo "cd server"
echo "source venv_server/bin/activate"
echo "python start_server.py"
echo
echo "To start the client (in new terminal):"
echo "cd client"
echo "source venv_client/bin/activate"
echo "python start_client.py"
echo
echo "Make sure to copy commands.json and models/ directory to the project root!"
echo
echo "Note: You may need to grant camera permissions in System Preferences > Security & Privacy > Camera"
echo

# Make the script executable
chmod +x setup_scripts/setup_macos.sh