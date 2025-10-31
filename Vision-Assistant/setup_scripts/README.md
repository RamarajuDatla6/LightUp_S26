# Setup Scripts

This directory contains automated setup scripts for different operating systems to make the Vision Assistant installation process easier.

## Available Scripts

### Windows
- **`setup_windows.bat`** - Automated setup for Windows systems

### Linux/Ubuntu
- **`setup_ubuntu.sh`** - Automated setup for Ubuntu/Debian systems

### macOS
- **`setup_macos.sh`** - Automated setup for macOS systems

## Usage

### Windows
```cmd
# Run the setup script
setup_scripts\setup_windows.bat
```

### Ubuntu/Linux
```bash
# Make script executable
chmod +x setup_scripts/setup_ubuntu.sh

# Run the setup script
./setup_scripts/setup_ubuntu.sh
```

### macOS
```bash
# Make script executable
chmod +x setup_scripts/setup_macos.sh

# Run the setup script
./setup_scripts/setup_macos.sh
```

## What the Scripts Do

1. **Check Prerequisites**: Verify Python installation and required tools
2. **Install System Dependencies**: Install OS-specific packages and libraries
3. **Create Virtual Environments**: Set up isolated Python environments for server and client
4. **Install Python Dependencies**: Install all required Python packages
5. **Provide Instructions**: Show next steps to run the application

## Prerequisites

### Windows
- Python 3.8+ installed and added to PATH
- Git installed
- Visual Studio Build Tools (for dlib compilation)

### Ubuntu/Linux
- Python 3.8+ installed
- sudo access for package installation

### macOS
- Python 3.8+ installed
- Homebrew (will be installed automatically if missing)

## Manual Setup

If you prefer to set up manually or the scripts don't work, refer to the main README.md file for detailed manual installation instructions.

## Troubleshooting

### Script Permission Issues (Linux/macOS)
```bash
chmod +x setup_scripts/*.sh
```

### Python Not Found
Make sure Python is installed and added to your system PATH.

### dlib Installation Issues (Windows)
Use conda instead of pip for easier dlib installation:
```bash
conda create -n vision_assistant python=3.9
conda activate vision_assistant
conda install -c conda-forge dlib
pip install -r server/requirements.txt
pip install -r client/requirements.txt
```

### Audio Issues (Linux)
Install additional audio dependencies:
```bash
sudo apt install -y pulseaudio pulseaudio-utils libasound2-dev
```

## After Setup

Once the setup is complete:

1. Copy `commands.json` to the project root
2. Copy the `models/` directory to the project root
3. Start the server: `cd server && source venv_server/bin/activate && python start_server.py`
4. Start the client: `cd client && source venv_client/bin/activate && python start_client.py` 