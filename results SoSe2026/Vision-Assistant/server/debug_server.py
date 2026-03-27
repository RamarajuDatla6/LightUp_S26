#!/usr/bin/env python3
"""
Debug script to test server functionality
"""

import requests
import base64
import cv2
import numpy as np
import json

SERVER_URL = "http://localhost:8000"

def test_server_health():
    """Test server health endpoint"""
    try:
        print("Testing server health...")
        response = requests.get(f"{SERVER_URL}/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            print("✓ Server is healthy")
            print(f"  Status: {health_data.get('status')}")
            print(f"  Models loaded: {health_data.get('models_loaded')}")
            print(f"  Known faces: {health_data.get('known_faces')}")
            return True
        else:
            print(f"✗ Server health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Cannot connect to server: {e}")
        return False

def create_test_frame():
    """Create a simple test frame"""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :] = [255, 0, 0]  # Blue frame
    cv2.putText(frame, "TEST FRAME", (200, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    return frame

def encode_frame(frame):
    """Encode frame to base64"""
    try:
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = base64.b64encode(buffer).decode('utf-8')
        return frame_base64
    except Exception as e:
        print(f"Error encoding frame: {e}")
        return None

def test_frame_processing():
    """Test frame processing endpoint"""
    print("\nTesting frame processing...")
    

    test_frame = create_test_frame()
    frame_b64 = encode_frame(test_frame)
    
    if frame_b64 is None:
        print("✗ Failed to encode test frame")
        return False
    
    print(f"Test frame encoded, length: {len(frame_b64)}")
    

    try:
        print("Testing request format...")
        response = requests.post(
            f"{SERVER_URL}/test_request",
            json={
                "frame_data": frame_b64,
                "mode": "object"
            },
            timeout=10.0
        )
        
        if response.status_code == 200:
            result = response.json()
            print("✓ Request format is valid")
            print(f"  Result: {result}")
        else:
            print(f"✗ Test request failed: {response.status_code}")
            print(f"  Response: {response.text}")
            return False
    except Exception as e:
        print(f"✗ Test request error: {e}")
        return False
    

    for mode in ['object', 'face', 'scene']:
        try:
            print(f"\nTesting {mode} mode...")
            response = requests.post(
                f"{SERVER_URL}/process_frame",
                json={
                    "frame_data": frame_b64,
                    "mode": mode
                },
                timeout=15.0
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    print(f"✓ {mode} mode works")
                    print(f"  Detection result: {result.get('detection_result')}")
                else:
                    print(f"✗ {mode} mode failed: {result.get('error')}")
            else:
                print(f"✗ {mode} mode request failed: {response.status_code}")
                print(f"  Response: {response.text}")
                
        except Exception as e:
            print(f"✗ {mode} mode error: {e}")
    
    return True

def test_speech():
    """Test speech endpoint"""
    print("\nTesting speech endpoint...")
    try:
        response = requests.post(
            f"{SERVER_URL}/speak",
            json={"text": "Testing speech functionality"},
            timeout=5.0
        )
        
        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                print("✓ Speech endpoint works")
            else:
                print(f"✗ Speech failed: {result.get('error')}")
        else:
            print(f"✗ Speech request failed: {response.status_code}")
            
    except Exception as e:
        print(f"✗ Speech error: {e}")

def main():
    print("Vision Assistant Server Debug Tool")
    print("=" * 40)
    

    if not test_server_health():
        print("\nServer is not running or not responding.")
        print("Please start the server first:")
        print("  cd server")
        print("  python start_server.py")
        return
    

    test_frame_processing()
    

    test_speech()
    
    print("\n" + "=" * 40)
    print("Debug tests completed!")

if __name__ == "__main__":
    main() 