# client.py

import streamlit as st
st.set_page_config(layout="wide", page_title="Vision Assistant")
import time
import cv2
import base64
import json
import requests
from battery_sim import get_system_stats
import threading
from speech_client import initialize_speech_recognition, process_speech, cleanup_speech_recognition
import voice_queue

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(layout="wide", page_title="Vision Assistant")

SERVER_URL = "http://localhost:8000"

def fetch_power_models():
    """Fetch available SoC power models from the server."""
    try:
        r = requests.get(f"{SERVER_URL}/power/models", timeout=3)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    # Fallback list so the UI still works if server is older
    return [
        {"id": "snapdragon_8_gen_3", "name": "Snapdragon 8 Gen 3", "notes": ""},
        {"id": "dimensity_9300", "name": "Dimensity 9300", "notes": ""},
        {"id": "iphone_17_a19", "name": "iPhone 17 (A19-class)", "notes": ""},
    ]

# -----------------------------------------------------
# SESSION STATE INIT (SAFE)
# -----------------------------------------------------
def init_state():
    if "initialized" in st.session_state:
        return
    
    st.session_state.initialized = True
    st.session_state.battery_percentage = None
    st.session_state.browser_cpu = None
    st.session_state.system_cpu = None  # Changed from python_cpu
    st.session_state.avg_cpu = None  # NEW: Smoothed average CPU
    st.session_state.cpu_trend = None  # NEW: CPU trend (increasing/stable/decreasing)
    st.session_state.runtime_hours = None
    st.session_state.browser_name = "Vision Assistant"
    st.session_state.last_update = 0.0
    st.session_state.mode = None
    st.session_state.messages = []
    st.session_state.last_spoken = {}
    st.session_state.cap = None
    
    # Voice recognition state
    st.session_state.voice_enabled = False
    st.session_state.speech_thread = None
    st.session_state.stop_speech_event = None
    st.session_state.recognizer = None
    st.session_state.audio_queue = None
    st.session_state.audio_stream = None

# -----------------------------------------------------
# VOICE COMMAND CALLBACK (THREAD-SAFE)
# -----------------------------------------------------
def handle_voice_command(action):
    """Handle voice commands by putting them in a thread-safe queue"""
    print(f"Voice command queued: {action}")
    voice_queue.put_command(action)

# -----------------------------------------------------
# PROCESS VOICE COMMANDS FROM QUEUE (MAIN THREAD)
# -----------------------------------------------------
def process_voice_commands():
    """Process any pending voice commands from the queue"""
    try:
        # Check if there are commands waiting
        queue_size = voice_queue.queue_size()
        if queue_size > 0:
            print(f"📥 Queue has {queue_size} command(s) waiting")
        
        while not voice_queue.is_empty():
            action = voice_queue.get_command()
            print(f"✅ Processing voice command from queue: {action}")
            
            if action == "object":
                st.session_state.mode = "object"
                st.session_state.messages.append(
                    f"{time.strftime('%H:%M:%S')} - Voice: Switched to Object Detection"
                )
                print(f"✅ Mode changed to: object")
            elif action == "face":
                st.session_state.mode = "face"
                st.session_state.messages.append(
                    f"{time.strftime('%H:%M:%S')} - Voice: Switched to Face Recognition"
                )
                print(f"✅ Mode changed to: face")
            elif action == "scene":
                st.session_state.mode = "scene"
                st.session_state.messages.append(
                    f"{time.strftime('%H:%M:%S')} - Voice: Switched to Scene Description"
                )
                print(f"✅ Mode changed to: scene")
            elif action == "stop":
                st.session_state.mode = None
                st.session_state.messages.append(
                    f"{time.strftime('%H:%M:%S')} - Voice: Stopped Detection"
                )
                print(f"✅ Mode changed to: None (stopped)")
    except Exception as e:
        print(f"❌ Error processing voice command: {e}")
        import traceback
        traceback.print_exc()

# -----------------------------------------------------
# VOICE RECOGNITION CONTROL
# -----------------------------------------------------
def start_voice_recognition():
    """Start voice recognition in background thread"""
    if st.session_state.voice_enabled:
        return
    
    try:
        recognizer, audio_queue, audio_callback, audio_stream = initialize_speech_recognition()
        
        if recognizer is None:
            st.error("Failed to initialize speech recognition")
            return
        
        st.session_state.recognizer = recognizer
        st.session_state.audio_queue = audio_queue
        st.session_state.audio_stream = audio_stream
        st.session_state.stop_speech_event = threading.Event()
        
        # Start audio stream
        audio_stream.start()
        
        # Start processing thread
        speech_thread = threading.Thread(
            target=process_speech,
            args=(recognizer, audio_queue, st.session_state.stop_speech_event, handle_voice_command),
            daemon=True
        )
        speech_thread.start()
        
        st.session_state.speech_thread = speech_thread
        st.session_state.voice_enabled = True
        
        st.session_state.messages.append(
            f"{time.strftime('%H:%M:%S')} - Voice Recognition: Started"
        )
        
    except Exception as e:
        st.error(f"Error starting voice recognition: {e}")

def stop_voice_recognition():
    """Stop voice recognition"""
    if not st.session_state.voice_enabled:
        return
    
    try:
        if st.session_state.stop_speech_event:
            st.session_state.stop_speech_event.set()
        
        if st.session_state.audio_stream:
            st.session_state.audio_stream.stop()
        
        cleanup_speech_recognition()
        
        st.session_state.voice_enabled = False
        st.session_state.messages.append(
            f"{time.strftime('%H:%M:%S')} - Voice Recognition: Stopped"
        )
        
    except Exception as e:
        st.error(f"Error stopping voice recognition: {e}")


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
    

# -----------------------------------------------------
# SYSTEM STATS UPDATE (ONCE PER SECOND)
# -----------------------------------------------------
def update_stats():
    now = time.time()
    if now - st.session_state.last_update < 1:
        return
    try:
        stats = get_system_stats()
        st.session_state.battery_percentage = stats.get("battery_percent")
        st.session_state.browser_cpu = stats.get("browser_cpu")
        st.session_state.system_cpu = stats.get("system_cpu")  # Total system CPU
        st.session_state.avg_cpu = stats.get("avg_cpu")  # Smoothed average
        st.session_state.cpu_trend = stats.get("cpu_trend")  # Trend indicator
        st.session_state.runtime_hours = stats.get("estimated_runtime_h")
        st.session_state.last_update = now
    except Exception as e:
        st.warning(f"System stats update failed: {e}")


@st.cache_data(ttl=30)
def fetch_power_models_cached():
    return fetch_power_models()
# -----------------------------------------------------
# MAIN APP
# -----------------------------------------------------
def main():
    init_state()
    update_stats()
    

    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Page", ["Vision Assistant", "Power Simulation"], index=0)

    # Fetch SoC profiles for simulator
    # models = fetch_power_models()
    models = fetch_power_models_cached()

    model_labels = {m["id"]: m.get("name", m["id"]) for m in models}
    model_ids = list(model_labels.keys())
    default_mid = st.session_state.get("power_model_id") or (model_ids[0] if model_ids else "snapdragon_8_gen_3")
    if default_mid not in model_ids and model_ids:
        default_mid = model_ids[0]
    st.sidebar.subheader("Power Simulation")
    selected_mid = st.sidebar.selectbox(
        "Mobile SoC profile",
        options=model_ids,
        format_func=lambda x: model_labels.get(x, x),
        index=(model_ids.index(default_mid) if default_mid in model_ids else 0),
    )
    selected_policy = st.sidebar.selectbox(
        "Policy",
        options=["balanced", "performance", "battery"],
        index=["balanced", "performance", "battery"].index(st.session_state.get("power_policy", "balanced")),
    )
    st.session_state["power_model_id"] = selected_mid
    st.session_state["power_policy"] = selected_policy

    # If the user is on the Power Simulation page, render it and exit early
    if page == "Power Simulation":
        st.title("📱 Mobile SoC Power Simulation")
        st.markdown("Estimate power/energy for Snapdragon 8 Gen 3, Dimensity 9300, and iPhone 17 (A19-class) profiles using an explainable model.")


        # Server required for /power/* endpoints
        server_healthy, health_data = check_server_health()
        if not server_healthy:
            st.error("Server is not running. Start the server first to use the simulator endpoints.")
            st.info("Run: `cd server && python start_server.py`")
            return

        with st.expander("Available SoC profiles", expanded=False):
            st.write(models)

        st.subheader("Workload-based estimate (steady-state)")
        presets = {
            "Object detection ~30 FPS": dict(fps=30.0, util_cpu=0.55, util_gpu=0.25, util_npu=0.0),
            "Face recognition ~15 FPS": dict(fps=15.0, util_cpu=0.45, util_gpu=0.05, util_npu=0.0),
            "Scene description (heavy)": dict(fps=3.0, util_cpu=0.75, util_gpu=0.05, util_npu=0.0),
            "NPU inference demo": dict(fps=30.0, util_cpu=0.25, util_gpu=0.05, util_npu=0.45),
        }
        preset_name = st.selectbox("Preset", options=list(presets.keys()), index=0)
        p = presets[preset_name]

        colA, colB = st.columns(2)
        with colA:
            fps = st.slider("FPS", 0.5, 120.0, float(p["fps"]), 0.5)
            util_cpu = st.slider("CPU utilization", 0.0, 1.0, float(p["util_cpu"]), 0.01)
            util_gpu = st.slider("GPU utilization", 0.0, 1.0, float(p["util_gpu"]), 0.01)
            util_npu = st.slider("NPU utilization", 0.0, 1.0, float(p["util_npu"]), 0.01)
        with colB:
            battery_wh = st.number_input("Battery (Wh)", min_value=5.0, max_value=40.0, value=float(health_data.get("battery_wh", 19.0)) if isinstance(health_data, dict) else 19.0)
            st.caption("Tip: typical phones are ~15–20 Wh. Set your device battery here for runtime estimate.")

        if st.button("Estimate workload power", use_container_width=True):
            try:
                r = requests.post(
                    f"{SERVER_URL}/power/workload",
                    json={
                        "model_id": st.session_state["power_model_id"],
                        "policy": st.session_state["power_policy"],
                        "fps": fps,
                        "util_cpu": util_cpu,
                        "util_gpu": util_gpu,
                        "util_npu": util_npu,
                        "battery_wh": battery_wh,
                    },
                    timeout=5,
                )
                if r.status_code == 200:
                    est = r.json()
                    st.session_state["last_power"] = est
                else:
                    st.error(f"Estimate failed: {r.status_code} {r.text}")
            except Exception as e:
                st.error(f"Estimate error: {e}")

        if st.session_state.get("last_power"):
            est = st.session_state["last_power"]
            m1, m2, m3 = st.columns(3)
            m1.metric("Avg power", f"{est.get('avg_power_mw', 0):.0f} mW")
            m2.metric("Runtime", f"{est.get('estimated_runtime_hours', '—')} h")
            m3.metric("Thermal capped", "Yes" if est.get("throttled") else "No")
            st.json(est)

        st.subheader("Timing-based estimate")
        st.caption("If you run Vision Assistant mode, the server returns timings_ms + a per-frame power estimate. You can also paste your timings here.")
        default_timings = st.session_state.get("last_timings_ms") or {"decode": 4.0, "inference": 25.0, "post": 6.0}
        timings_text = st.text_area("timings_ms JSON", value=json.dumps(default_timings, indent=2), height=120)
        mode_for_timings = st.selectbox("Mode", ["object", "face", "scene"], index=0)


        if st.button("Estimate from timings", use_container_width=True):
            try:
                timings = json.loads(timings_text)

                payload = {
                    "model_id": st.session_state["power_model_id"],
                    "policy": st.session_state["power_policy"],
                    "mode": mode_for_timings,
                    "timings_ms": timings,
                    "include_overhead": True,
                    "overhead_mw": 250.0,
                }

                r = requests.post(f"{SERVER_URL}/power/estimate", json=payload, timeout=8)

                st.session_state["last_request_payload"] = payload
                st.session_state["last_response_status"] = r.status_code
                st.session_state["last_response_text"] = r.text

                if r.status_code == 200:
                    est = r.json()
                    st.session_state["last_power"] = est
                    st.success("✅ Estimate received from server!")
                else:
                    st.error(f"❌ Estimate failed: {r.status_code}")
            except Exception as e:
                st.error(f"❌ Timing estimate error: {e}")

        if st.session_state.get("last_response_status") is not None:
            st.markdown("### 🔎 Debug (last estimate call)")
            st.write("Status:", st.session_state.get("last_response_status"))
            st.code(st.session_state.get("last_request_payload", {}), language="json")
            st.text_area("Raw response (server)", st.session_state.get("last_response_text", ""), height=140)

        if st.session_state.get("last_power"):
            st.markdown("### ✅ Parsed estimate (JSON)")
            st.json(st.session_state["last_power"])



        return


    # Process any pending voice commands (thread-safe)
    # This runs every time Streamlit reruns the app
    current_mode = st.session_state.get('mode', 'None')
    queue_size = voice_queue.queue_size()
    # print(f"🔄 Main loop - Mode: {current_mode}, Queue size: {queue_size}")
    process_voice_commands()
    
    # -------------------------------------------------
    # TITLE
    # -------------------------------------------------
    st.title("🎧 Voice-Only Vision Assistant for Blind People")
    st.markdown(
        "### Audio-based object detection, face recognition, and scene description"
    )
    
    # -------------------------------------------------
    # SAFE VALUES
    # -------------------------------------------------
    battery = st.session_state.battery_percentage
    browser_cpu = st.session_state.browser_cpu
    system_cpu = st.session_state.system_cpu
    avg_cpu = st.session_state.avg_cpu
    cpu_trend = st.session_state.cpu_trend
    runtime = st.session_state.runtime_hours
    
    battery_text = f"{battery:.2f}%" if battery is not None else "N/A"
    browser_cpu_text = f"{browser_cpu:.1f}%" if browser_cpu is not None else "N/A"
    system_cpu_text = f"{system_cpu:.1f}%" if system_cpu is not None else "N/A"
    avg_cpu_text = f"{avg_cpu:.1f}%" if avg_cpu is not None else "N/A"
    
    # Trend icon
    trend_icons = {
        "increasing": "📈",
        "stable": "➡️",
        "decreasing": "📉"
    }
    trend_icon = trend_icons.get(cpu_trend, "➡️")
    trend_display = cpu_trend.title() if cpu_trend else "N/A"
    
    runtime_text = (
        f"{runtime:.2f} hours remaining"
        if runtime is not None
        else "Calculating..."
    )
    
    # -------------------------------------------------
    # STATUS BAR (UPDATED WITH ALL METRICS)
    # -------------------------------------------------
    voice_status = "🎤 ACTIVE" if st.session_state.voice_enabled else "🎤 INACTIVE"
    voice_color = "#00ff00" if st.session_state.voice_enabled else "#ff0000"
    
    st.markdown(
        f"""
        <div style='border:2px solid #222; border-radius:8px; padding:12px; margin-bottom:12px;'>
            <b>🔋 Battery:</b> {battery_text} &nbsp;&nbsp;
            <b>🌐 Browser Tab:</b> {browser_cpu_text} &nbsp;&nbsp;
            <b>💻 System CPU:</b> {system_cpu_text} &nbsp;&nbsp;
            <b>📊 Avg CPU (10s):</b> {avg_cpu_text} &nbsp;&nbsp;
            <b>{trend_icon} Trend:</b> {trend_display} &nbsp;&nbsp;
            <b>⏱️ Runtime:</b> {runtime_text} &nbsp;&nbsp;
            <b style='color:{voice_color}'>Voice: {voice_status}</b>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if battery is not None and battery <= 12:
        st.warning("⚠️ Low battery detected. Please consider charging the device.")
    
    # -------------------------------------------------
    # CONTROL PANEL
    # -------------------------------------------------
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🎛️ Control Panel")
        
        # Voice Control Toggle
        st.markdown("#### 🎤 Voice Control")
        voice_col1, voice_col2 = st.columns(2)
        
        with voice_col1:
            if st.button("▶️ Start Voice", use_container_width=True):
                start_voice_recognition()
        
        with voice_col2:
            if st.button("⏸️ Stop Voice", use_container_width=True):
                stop_voice_recognition()
        
        st.markdown("---")
        
        # Mode Selection
        st.markdown("#### 🎯 Detection Mode")
        
        if st.session_state.mode:
            st.info(f"Active Mode: {st.session_state.mode.title()}")
        else:
            st.info("No detection active")
        
        if st.button("🎯 Object Detection", use_container_width=True):
            st.session_state.mode = "object"
            st.session_state.messages.append(
                f"{time.strftime('%H:%M:%S')} - Button: Switched to Object Detection"
            )
        
        if st.button("👤 Face Recognition", use_container_width=True):
            st.session_state.mode = "face"
            st.session_state.messages.append(
                f"{time.strftime('%H:%M:%S')} - Button: Switched to Face Recognition"
            )
        
        if st.button("🖼️ Scene Description", use_container_width=True):
            st.session_state.mode = "scene"
            st.session_state.messages.append(
                f"{time.strftime('%H:%M:%S')} - Button: Switched to Scene Description"
            )

        if st.button("🧠 Isaac VLM (Vision-Language)", use_container_width=True):
            st.session_state.mode = "isaac"
            st.session_state.messages.append(
                f"{time.strftime('%H:%M:%S')} - Button: Switched to Isaac VLM"
            )
        st.markdown("#### 🧠 Isaac VLM Settings")
        st.session_state["isaac_prompt"] = st.text_area(
            "Isaac prompt",
            value=st.session_state.get(
                "isaac_prompt",
                "Describe the scene for a blind user in 1-2 short sentences. Mention people, objects, and obstacles.",
            ),
            height=90,
        )

        # st.session_state["isaac_hint"] = st.selectbox(
        #     "Isaac hint",
        #     options=["NONE", "BOX", "POINT", "POLYGON"],
        #     index=["NONE", "BOX", "POINT", "POLYGON"].index(st.session_state.get("isaac_hint", "NONE")),
        # )


        
        if st.button("⏹️ Stop Detection", use_container_width=True):
            st.session_state.mode = None
            st.session_state.messages.append(
                f"{time.strftime('%H:%M:%S')} - Button: Stopped Detection"
            )
        

    
    # -------------------------------------------------
    # MESSAGE LOG & CPU INFO
    # -------------------------------------------------
    with col2:
        st.markdown("### 📝 Audio Messages")
        
        if st.session_state.messages:
            st.text_area(
                "Recent Messages",
                "\n".join(st.session_state.messages[-15:]),
                height=300,
                disabled=True
            )
        else:
            st.info("No messages yet")
        
        # CPU Metrics Display
        st.markdown("---")
        st.markdown("### 📊 CPU Metrics")
        
        metric_col1, metric_col2 = st.columns(2)
        
        with metric_col1:
            st.metric(
                label="Current System CPU",
                value=system_cpu_text,
                delta=None
            )
            st.metric(
                label="Browser Tab CPU",
                value=browser_cpu_text,
                delta=None
            )
        
        with metric_col2:
            st.metric(
                label="10s Average CPU",
                value=avg_cpu_text,
                delta=None
            )
            st.metric(
                label="CPU Trend",
                value=trend_display,
                delta=trend_icon
            )
        
        # Voice Commands Help
        if st.session_state.voice_enabled:
            st.markdown("---")
            st.markdown("#### 🗣️ Voice Commands")
            st.markdown("""
            - **"object detection"** - Start object detection
            - **"face recognition"** - Start face recognition  
            - **"scene description"** - Start scene description
            - **"stop detection"** - Stop current mode
            """)
    
    # -------------------------------------------------
    # CAMERA SETUP
    # -------------------------------------------------
    if st.session_state.cap is None:
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        st.session_state.cap = cap
    
    cap = st.session_state.cap
    
    # -------------------------------------------------
    # FRAME PROCESSING
    # -------------------------------------------------
    # if cap.isOpened() and st.session_state.mode:
    #     ret, frame = cap.read()
    #     if ret:
    #         _, buffer = cv2.imencode(".jpg", frame)
    #         frame_b64 = base64.b64encode(buffer).decode("utf-8")
            
    #         try:
    #             response = requests.post(
    #                 f"{SERVER_URL}/process_frame",
    #                 json={
    #                     "frame_data": frame_b64,
    #                     "mode": st.session_state.mode,
    #                     "model_id": st.session_state["power_model_id"],
    #                     "policy": st.session_state["power_policy"],
    #                 },

    #                 timeout=5
    #             )
                
    #             if response.status_code == 200:
    #                 result = response.json()
    #                 if result.get("success"):
    #                     detection = result.get("detection_result")
                        
    #                     if detection:
    #                         now = time.time()
    #                         mode = st.session_state.mode
                            
    #                         if now - st.session_state.last_spoken.get(mode, 0) >= 3:
    #                             if mode == "face":
    #                                 if isinstance(detection, list):
    #                                     msg = f"Recognized: {', '.join(detection)}"
    #                                 else:
    #                                     msg = f"Recognized: {detection}"
    #                             elif mode == "object":
    #                                 if isinstance(detection, list):
    #                                     msg = f"I see {', '.join(detection)}"
    #                                 else:
    #                                     msg = f"I see {detection}"
    #                             elif mode == "isaac":
    #                                     msg = f"Isaac says: {detection}"

    #                             else:
    #                                 msg = f"Scene description: {detection}"
                                
    #                             requests.post(
    #                                 f"{SERVER_URL}/speak",
    #                                 json={"text": msg}
    #                             )
    #                             st.session_state.last_spoken[mode] = now
    #                             st.session_state.messages.append(
    #                                 f"{time.strftime('%H:%M:%S')} - {msg}"
    #                             )
    #         except Exception as e:
    #             pass

    if cap.isOpened() and st.session_state.mode:
        ret, frame = cap.read()
        if ret:
            _, buffer = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buffer).decode("utf-8")

            # Gate Isaac calls (heavy model)
            if "last_isaac_call" not in st.session_state:
                st.session_state.last_isaac_call = 0.0

            if st.session_state.mode == "isaac":
                if time.time() - st.session_state.last_isaac_call < 3.5:
                    # skip extra isaac requests
                    pass
                else:
                    st.session_state.last_isaac_call = time.time()

            payload = {
                "frame_data": frame_b64,
                "mode": st.session_state.mode,
                "model_id": st.session_state["power_model_id"],
                "policy": st.session_state["power_policy"],
            }

            if st.session_state.mode == "isaac":
                payload["vlm_prompt"] = st.session_state.get("isaac_prompt")
                # payload["vlm_hint"] = st.session_state.get("isaac_hint", "NONE")


            timeout_s = 30 if st.session_state.mode == "isaac" else 5

            try:
                response = requests.post(f"{SERVER_URL}/process_frame", json=payload, timeout=timeout_s)

                if response.status_code == 200:
                    result = response.json()
                    if result.get("success"):
                        detection = result.get("detection_result")

                        # Show last result in UI
                        st.session_state["last_detection"] = detection

                        if detection:
                            # detection = result.get("detection_result")
                            now = time.time()
                            mode = st.session_state.mode
                            speak_interval = 6 if mode == "isaac" else 3

                            if now - st.session_state.last_spoken.get(mode, 0) >= speak_interval:
                                if mode == "face":
                                    msg = f"Recognized: {', '.join(detection)}" if isinstance(detection, list) else f"Recognized: {detection}"
                                elif mode == "object":
                                    msg = f"I see {', '.join(detection)}" if isinstance(detection, list) else f"I see {detection}"
                                elif mode == "isaac":
                                    msg = f"Isaac says: {detection}"
                                else:
                                    msg = f"Scene description: {detection}"

                                requests.post(f"{SERVER_URL}/speak", json={"text": msg}, timeout=10)
                                st.session_state.last_spoken[mode] = now
                                st.session_state.messages.append(f"{time.strftime('%H:%M:%S')} - {msg}")
                else:
                    st.session_state.messages.append(f"{time.strftime('%H:%M:%S')} - HTTP {response.status_code}: {response.text}")

            except Exception as e:
                st.session_state.messages.append(f"{time.strftime('%H:%M:%S')} - ERROR: {e}")


    
    # -------------------------------------------------
    # CONTROLLED REFRESH (SAFE)
    # -------------------------------------------------
    # Only auto-refresh in Vision Assistant live mode
    if page == "Vision Assistant":
        time.sleep(0.08)
        st.rerun()


# -----------------------------------------------------
# ENTRY POINT
# -----------------------------------------------------
if __name__ == "__main__":
    main()
