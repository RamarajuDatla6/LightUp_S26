import streamlit as st
# Configure Streamlit page
st.set_page_config(layout="wide", page_title="Vision Assistant")


import cv2
import numpy as np
import time
import threading
import base64
import requests
import json
import psutil
import random
from battery_sim import (
    smart_battery_drain,
    exponential_battery_drain,
    cpu_based_battery_drain,
    temperature_based_battery_drain,
    random_event_drain
)

from speech_client import (
    initialize_speech_recognition, 
    process_speech, 
    cleanup_speech_recognition
)

# Server configuration (NOTE: This must be updated based on server IP and port on which the backend is running)
SERVER_URL = "http://localhost:8000"

class VisionApp:
    def __init__(self):
        pass

    def encode_frame(self, frame):
        """Encode frame to base64 for sending to server for faster and better performance"""
        try:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return frame_base64
        except Exception as e:
            print(f"Error encoding frame: {e}")
            return None

    def process_frame_with_server(self, frame, mode):

        try:
            print(f"\n=== CLIENT REQUEST DEBUG ===")
            print(f"Mode: {mode} (type: {type(mode).__name__})")
            print(f"Frame shape: {frame.shape}")
            
            # Validate mode
            if mode is None:
                print("Mode is None - skipping processing")
                return None
                
            if mode not in ['face', 'object', 'scene']:
                print(f"Invalid mode: {mode} - must be one of: face, object, scene")
                return None
            
            print(f"Mode is valid: {mode}")
            
            # Encode frame
            frame_b64 = self.encode_frame(frame)
            if frame_b64 is None:
                print(" Failed to encode frame")
                return None

            print(f" Frame encoded, length: {len(frame_b64)}")
            
            # Prepare request data
            request_data = {
                "frame_data": frame_b64,
                "mode": mode
            }

            # Send request to server
            print(f"Sending POST request to: {SERVER_URL}/process_frame")
            response = requests.post(
                f"{SERVER_URL}/process_frame",
                json=request_data,
                timeout=5.0
            )

            print(f"Response status code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f" Server response successful")
                print(f"Response keys: {list(result.keys())}")
                
                if result.get("success"):
                    # Return only detection results - no visual frame
                    return result.get("detection_result")
                else:
                    print(f" Server processing error: {result.get('error')}")
                    return None
            else:
                print(f" Server request failed: {response.status_code}")
                try:
                    error_content = response.text
                    print(f"Error response content: {error_content}")
                except Exception as e:
                    print(f"Could not parse error response: {e}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"Error communicating with server: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error in frame processing: {e}")
            return None

    def speak_via_server(self, text):
        """Send text-to-speech request to server"""
        try:
            response = requests.post(
                f"{SERVER_URL}/speak",
                json={"text": text},
                timeout=1.0
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Error with server TTS: {e}")
            return False

def check_server_health():
    """Check if server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health", timeout=5.0)
        if response.status_code == 200:
            health_data = response.json()
            return health_data.get("status") == "healthy", health_data
        return False, None
    except Exception as e:
        return False, str(e)

# Initialize battery and timer in session state
if 'battery_percentage' not in st.session_state:
    st.session_state['battery_percentage'] = 100.0
if 'last_battery_update' not in st.session_state:
    st.session_state['last_battery_update'] = time.time()
if 'cpu_usage' not in st.session_state:
    st.session_state['cpu_usage'] = 0.0
if 'temperature' not in st.session_state:
    st.session_state['temperature'] = 40.0
if 'low_battery_spoken' not in st.session_state:
    st.session_state['low_battery_spoken'] = False

# Battery update every second
usage_rate = 0.01
lambda_value = 0.005
drain_factor = 0.02
temperature_factor = 0.01

def update_battery_every_second():
    now = time.time()
    if now - st.session_state['last_battery_update'] >= 1:
        battery_percentage = st.session_state['battery_percentage']
        cpu_usage = psutil.cpu_percent(interval=None)
        temperature = random.uniform(40, 60)
        

        st.session_state['cpu_usage'] = cpu_usage
        st.session_state['temperature'] = temperature
        
        battery_percentage = smart_battery_drain(usage_rate, battery_percentage, 1)
        battery_percentage = exponential_battery_drain(battery_percentage, lambda_value, 1)
        battery_percentage = cpu_based_battery_drain(cpu_usage, battery_percentage, 1, drain_factor)
        battery_percentage = temperature_based_battery_drain(temperature, battery_percentage, 1, temperature_factor)
        battery_percentage = random_event_drain(battery_percentage)
        
        # Speak low battery warning once
        if battery_percentage <= 12 and not st.session_state['low_battery_spoken']:
            try:
                requests.post(f"{SERVER_URL}/speak", json={"text": "Battery is low"})
                st.session_state['low_battery_spoken'] = True
            except Exception as e:
                print(f"Error speaking low battery warning: {e}")
        
        if battery_percentage <= 10:
            battery_percentage = 100.0
            st.session_state['low_battery_spoken'] = False  # Reset flag when battery recharges
            
        st.session_state['battery_percentage'] = battery_percentage
        st.session_state['last_battery_update'] = now
        st.rerun()

def main():
    # Initialize session state variables
    if 'mode' not in st.session_state:
        st.session_state.mode = None
    if 'stop_signal' not in st.session_state:
        st.session_state.stop_signal = False
    if 'voice_command_active' not in st.session_state:
        st.session_state.voice_command_active = False
    if 'speech_stop_event' not in st.session_state:
        st.session_state.speech_stop_event = threading.Event()
    if 'speech_thread' not in st.session_state:
        st.session_state.speech_thread = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'last_spoken' not in st.session_state:
        st.session_state.last_spoken = {}
    if 'cap' not in st.session_state:
        st.session_state.cap = None

    # Update battery every second
    update_battery_every_second()

    st.title("🎧 Voice-Only Vision Assistant for Blind People")
    st.markdown("### Audio-based object detection, face recognition, and scene description")

    # Battery, CPU, and Temp status box (using session state values)
    battery_percentage = st.session_state['battery_percentage']
    cpu_usage = st.session_state['cpu_usage']
    temperature = st.session_state['temperature']
    
    # Show warning if battery is low
    if battery_percentage <= 12:
        st.warning(f" Low Battery Warning: {battery_percentage:.2f}% - Please conserve power!")
    
    st.markdown(f"""
    <div style='border:2px solid #222; border-radius:8px; padding:12px; margin-bottom:12px; width:100%;'>
        <b>🔋 Battery:</b> {battery_percentage:.2f}% &nbsp;&nbsp; <b>🖥️ CPU:</b> {cpu_usage:.1f}% &nbsp;&nbsp; <b>🌡️ Temp:</b> {temperature:.1f}°C
    </div>
    """, unsafe_allow_html=True)

    # Check server health
    server_healthy, health_data = check_server_health()
    if not server_healthy:
        st.error(f" Server is not running or unhealthy. Please start the server first.")
        st.info("To start the server, run: `cd server && python server.py`")
        return




    col1, col2 = st.columns([1, 1])

    # Control Panel
    with col1:
        st.markdown("### 🎛️ Control Panel")
        
        # Show current mode
        if st.session_state.mode:
            st.info(f"🔄 Active: {st.session_state.mode.title()} Detection")
        else:
            st.info("⏸️ No detection active")

        # Voice command toggle button
        voice_command_text = "🎤 Stop Voice Commands" if st.session_state.voice_command_active else "🎤 Start Voice Commands"
        if st.button(voice_command_text, key="voice_cmd_btn", use_container_width=True):
            try:
                if not st.session_state.voice_command_active:
                    st.info("Starting voice commands...")
                    components = initialize_speech_recognition()
                    if None in components:
                        st.error("Failed to initialize speech recognition")
                        return

                    recognizer, audio_queue, _, stream = components
                    st.session_state.speech_stop_event.clear()
                    stream.start()

                    def mode_callback(mode):
                        st.session_state.mode = mode

                    st.session_state.speech_thread = threading.Thread(
                        target=process_speech,
                        args=(recognizer, audio_queue, st.session_state.speech_stop_event, mode_callback)
                    )
                    st.session_state.speech_thread.daemon = True
                    st.session_state.speech_thread.start()

                    st.session_state.voice_command_active = True
                    requests.post(f"{SERVER_URL}/speak", json={"text": "Voice commands activated"})
                    st.success("Voice commands are now active")
                else:
                    st.info("Stopping voice commands...")
                    st.session_state.voice_command_active = False
                    st.session_state.speech_stop_event.set()
                    cleanup_speech_recognition()
                    if st.session_state.speech_thread:
                        st.session_state.speech_thread.join(timeout=1)
                        st.session_state.speech_thread = None
                    requests.post(f"{SERVER_URL}/speak", json={"text": "Voice commands deactivated"})
                    st.warning("Voice commands are now inactive")

            except Exception as e:
                print(f"Error toggling voice commands: {e}")
                st.error(f"Error toggling voice commands: {str(e)}")
            st.rerun()

        st.markdown("---")

        # Detection mode buttons
        if st.button("🎯 Object Detection", key="object_detect_btn", use_container_width=True):
            st.session_state.mode = 'object'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting object detection"})
            st.rerun()

        if st.button("👤 Face Recognition", key="face_recog_btn", use_container_width=True):
            st.session_state.mode = 'face'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting face recognition"})
            st.rerun()

        if st.button("🖼️ Scene Description", key="scene_desc_btn", use_container_width=True):
            st.session_state.mode = 'scene'
            st.session_state.stop_signal = False
            requests.post(f"{SERVER_URL}/speak", json={"text": "Starting scene description"})
            st.rerun()

        if st.button("⏹️ Stop Detection", key="stop_btn", use_container_width=True):
            st.session_state.mode = None
            st.session_state.stop_signal = True
            requests.post(f"{SERVER_URL}/speak", json={"text": "Stopping detection"})
            st.rerun()

    # Messages Panel
    with col2:
        st.markdown("### 📝 Audio Messages Log")
        
        if st.session_state.messages:

            messages_text = "\n".join([f"• {msg}" for msg in reversed(st.session_state.messages[-10:])])
            st.text_area("Recent Audio Announcements", messages_text, height=400, disabled=True, key="messages_area")
        else:
            st.info("No messages yet. Start a detection mode to begin.")

        # Clear messages button
        if st.button("🗑️ Clear Messages", key="clear_msgs_btn"):
            st.session_state.messages = []
            st.rerun()

    # Initialize camera
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera")
            st.stop()
        # Configure camera settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        st.session_state.cap = cap

    # Background processing - capture and analyze frames without display
    if st.session_state.cap is not None and st.session_state.cap.isOpened():
        try:
            ret, frame = st.session_state.cap.read()
            if not ret:
                st.error("Failed to read from camera")
                return

            # Process frame based on mode - only if mode is set and not None
            if st.session_state.mode and st.session_state.mode in ['face', 'object', 'scene']:

                app = VisionApp()
                detection_result = app.process_frame_with_server(frame, st.session_state.mode)

                if detection_result:
                    current_time = time.time()
                    mode_key = st.session_state.mode
                    
                    # Only speak if enough time has passed since last announcement
                    if mode_key not in st.session_state.last_spoken or \
                       current_time - st.session_state.last_spoken.get(mode_key, 0) >= 3.0:
                        if st.session_state.mode == 'face':
                            message = f"Recognized: {', '.join(detection_result)}"
                        elif st.session_state.mode == 'object':
                            message = f"I see {', '.join(detection_result)}"
                        else:  # scene mode
                            message = f"Scene: {detection_result}"
                        
                        requests.post(f"{SERVER_URL}/speak", json={"text": message})
                        st.session_state.last_spoken[mode_key] = current_time
                        st.session_state.messages.append(f"{time.strftime('%H:%M:%S')} - {message}")

            # Control processing rate
            time.sleep(0.1)  # Process ~10 times per second

        except Exception as e:
            st.error(f"Error processing frame: {str(e)}")
            print(f"Frame processing error: {e}")

        # Continuous refresh for background processing - only if not stopped
        if not st.session_state.stop_signal:
            st.rerun()

if __name__ == "__main__":
    main()