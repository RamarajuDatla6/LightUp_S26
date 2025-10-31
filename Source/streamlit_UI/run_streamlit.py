import streamlit as st
import time

# Configure Streamlit page - must be first Streamlit command
st.set_page_config(layout="wide", page_title="Vision Assistant - Static UI")


def main():
    # Static data for UI demonstration
    battery_percentage = 75.5
    cpu_usage = 24.3
    temperature = 45.2
    current_mode = "object"  # Options: None, "object", "face", "scene"
    voice_commands_active = True
    server_healthy = True

    # Simulated log messages for the interface
    static_messages = [
        "14:23:15 - Scene: A person sitting at a desk with a laptop computer",
        "14:23:10 - I see laptop, cup, keyboard, mouse",
        "14:23:05 - Recognized: John Smith",
        "14:23:00 - Starting face recognition",
        "14:22:55 - I see book, pen, notebook, phone",
        "14:22:50 - Scene: A cluttered office desk with various office supplies",
        "14:22:45 - Starting scene description",
        "14:22:40 - I see chair, desk, monitor",
        "14:22:35 - Recognized: Unknown person",
        "14:22:30 - Starting object detection"
    ]
    # Application header and description
    st.title(" Voice-Only Vision Assistant for Blind People")
    st.markdown("### Audio-based object detection, face recognition, and scene description")

    # Display warning if battery is below a threshold
    if battery_percentage <= 12:
        st.warning(f" Low Battery Warning: {battery_percentage:.2f}% - Please conserve power!")

    # Display system status: battery, CPU usage, and temperature
    st.markdown(f"""
    <div style='border:2px solid #222; border-radius:8px; padding:12px; margin-bottom:12px; width:100%;'>
        <b> Battery:</b> {battery_percentage:.2f}% &nbsp;&nbsp; <b>🖥️ CPU:</b> {cpu_usage:.1f}% &nbsp;&nbsp; <b>🌡️ Temp:</b> {temperature:.1f}°C
    </div>
    """, unsafe_allow_html=True)

    # Show server health status
    if server_healthy:
        st.success(" Server is running and healthy")
    else:
        st.error(f" Server is not running or unhealthy. Please start the server first.")
        st.info("To start the server, run: `cd server && python server.py`")

    # two columns for layout
    col1, col2 = st.columns([1, 1])

    # Left column: Control Panel
    with col1:
        st.markdown("###  Control Panel")

        # Show which detection mode is currently active
        if current_mode:
            st.info(f" Active: {current_mode.title()} Detection")
        else:
            st.info("⏸ No detection active")

        # Button to toggle voice command system
        voice_command_text = "Stop Voice Commands" if voice_commands_active else " Start Voice Commands"
        if st.button(voice_command_text, key="voice_cmd_btn", use_container_width=True):
            if voice_commands_active:
                st.success("Voice commands activated")
            else:
                st.warning("Voice commands deactivated")

        st.markdown("---")

        # Buttons to start different detection modes
        if st.button("🎯 Object Detection", key="object_detect_btn", use_container_width=True):
            st.success("Object detection started")

        if st.button("👤 Face Recognition", key="face_recog_btn", use_container_width=True):
            st.success("Face recognition started")

        if st.button("🖼️ Scene Description", key="scene_desc_btn", use_container_width=True):
            st.success("Scene description started")

        if st.button("⏹️ Stop Detection", key="stop_btn", use_container_width=True):
            st.info("Detection stopped")

    # Messages Panel
    with col2:
        st.markdown("### 📝 Audio Messages Log")

        # Display static messages in reverse order
        messages_text = "\n".join([f"• {msg}" for msg in reversed(static_messages[-10:])])
        st.text_area("Recent Audio Announcements", messages_text, height=400, disabled=True, key="messages_area")

        # Button to simulate clearing the messages
        if st.button("🗑️ Clear Messages", key="clear_msgs_btn"):
            st.info("Messages cleared")



# Run the main function if script is executed
if __name__ == "__main__":
    main()