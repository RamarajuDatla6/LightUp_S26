#!/usr/bin/env python3
"""
Automatic model downloader for Vision Assistant
"""

import os
import urllib.request
import zipfile
import tarfile
import bz2
import shutil
from pathlib import Path

def download_file(url, filename, description=""):
    """Download a file with progress indication"""
    print(f"📥 Downloading {description}...")
    print(f"   URL: {url}")
    print(f"   File: {filename}")
    
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"✅ Downloaded: {filename}")
        return True
    except Exception as e:
        print(f"❌ Failed to download {filename}: {e}")
        return False

def extract_bz2(filename, target):
    """Extract bz2 file"""
    print(f"📦 Extracting {filename}...")
    try:
        with bz2.open(filename, 'rb') as f_in:
            with open(target, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        os.remove(filename)  # Remove compressed file
        print(f"✅ Extracted: {target}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")
        return False

def extract_zip(filename, target_dir):
    """Extract zip file"""
    print(f"📦 Extracting {filename}...")
    try:
        with zipfile.ZipFile(filename, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        os.remove(filename)  # Remove zip file
        print(f"✅ Extracted to: {target_dir}")
        return True
    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")
        return False

def create_directories():
    """Create required directories"""
    print("📁 Creating directories...")
    
    directories = [
        "models",
        "models/speech",
        "models/speech/model", 
        "known_faces"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}/")

def download_face_models():
    """Download face recognition models"""
    print("\n🔍 DOWNLOADING FACE RECOGNITION MODELS")
    print("=" * 50)
    
    models = [
        {
            "url": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
            "file": "models/deploy.prototxt",
            "desc": "Face Detection Configuration"
        },
        {
            "url": "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
            "file": "models/res10_300x300_ssd_iter_140000.caffemodel",
            "desc": "Face Detection Model"
        },
        {
            "url": "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2",
            "file": "models/shape_predictor_68_face_landmarks.dat.bz2",
            "extract_to": "models/shape_predictor_68_face_landmarks.dat",
            "desc": "Facial Landmarks Predictor"
        },
        {
            "url": "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2", 
            "file": "models/dlib_face_recognition_resnet_model_v1.dat.bz2",
            "extract_to": "models/dlib_face_recognition_resnet_model_v1.dat",
            "desc": "Face Recognition Model"
        }
    ]
    
    for model in models:
        if os.path.exists(model.get("extract_to", model["file"].replace(".bz2", ""))):
            print(f"⏭️  Already exists: {model['desc']}")
            continue
            
        if download_file(model["url"], model["file"], model["desc"]):
            if model["file"].endswith(".bz2"):
                extract_bz2(model["file"], model["extract_to"])

def download_speech_model():
    """Download speech recognition model"""
    print("\n🎤 DOWNLOADING SPEECH RECOGNITION MODEL")
    print("=" * 50)
    
    model_dir = "models/speech/model/vosk-model-small-en-us-0.15"
    if os.path.exists(model_dir):
        print("⏭️  Speech model already exists")
        return
    
    url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    filename = "models/speech/model/vosk-model-small-en-us-0.15.zip"
    
    if download_file(url, filename, "Vosk Speech Recognition Model"):
        extract_zip(filename, "models/speech/model")

def check_existing_models():
    """Check which models already exist"""
    print("\n🔍 CHECKING EXISTING MODELS")
    print("=" * 50)
    
    required_files = [
        "models/deploy.prototxt",
        "models/res10_300x300_ssd_iter_140000.caffemodel", 
        "models/shape_predictor_68_face_landmarks.dat",
        "models/dlib_face_recognition_resnet_model_v1.dat",
        "models/speech/model/vosk-model-small-en-us-0.15"
    ]
    
    existing = []
    missing = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            existing.append(file_path)
            print(f"✅ Found: {file_path}")
        else:
            missing.append(file_path)
            print(f"❌ Missing: {file_path}")
    
    print(f"\n📊 Summary: {len(existing)}/{len(required_files)} models found")
    return missing

def main():
    print("🤖 Vision Assistant Model Downloader")
    print("=" * 50)
    print("This script will download all required AI models")
    print("Total download size: ~500MB")
    print("=" * 50)
    
    # Check current directory
    if not os.path.exists("server") or not os.path.exists("client"):
        print("❌ Please run this script from the project root directory")
        print("   (The directory containing server/ and client/ folders)")
        return
    
    # Create directories
    create_directories()
    
    # Check existing models
    missing = check_existing_models()
    
    if not missing:
        print("\n🎉 All models already downloaded!")
        return
    
    print(f"\n📥 Need to download {len(missing)} models...")
    
    # Download models
    download_face_models()
    download_speech_model()
    
    print("\n🎉 MODEL DOWNLOAD COMPLETE!")
    print("=" * 50)
    print("✅ All required models have been downloaded")
    print("✅ You can now start the Vision Assistant server")
    print("\nNext steps:")
    print("1. cd server")
    print("2. python start_server.py")
    print("3. In another terminal: cd client && python start_client.py")
    
    # Final check
    print("\n🔍 FINAL MODEL CHECK")
    print("=" * 30)
    check_existing_models()

if __name__ == "__main__":
    main() 