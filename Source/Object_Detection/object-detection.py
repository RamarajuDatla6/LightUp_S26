from ultralytics import YOLO
import cv2
import requests
import time
import os
import uuid
import simpleaudio as sa
import errno

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Init webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Control
frame_count = 0
frame_skip = 10  # Process every 10th frame for efficiency
cooldown_seconds = 5
last_alert_time_by_label = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue # Skip frames for performance

    results = model.predict(frame, conf=0.5, verbose=False)
    names = model.names

    for r in results:
        for cls in r.boxes.cls:
            label = names[int(cls)]
            print("Detected:", label)

            now = time.time()
            last_time = last_alert_time_by_label.get(label, 0)

            if now - last_time > cooldown_seconds:
                print(f"Triggering TTS for: {label}")
                # Send POST request to TTS server to generate speech
                response = requests.post(
                    "http://localhost:8000/speak",
                    # data={"text": f"Vrushabh has been detected!"}
                    data={"text": f"A {label} has been detected!"}
                )

                if response.ok:
                    try:
                        # Unique file name per label
                        filename = f"alert_{label.replace(' ', '_')}_{uuid.uuid4().hex}.wav"

                        # Save TTS audio
                        with open(filename, "wb") as f:
                            f.write(response.content)

                        # Play the saved file using simpleaudio
                        wave_obj = sa.WaveObject.from_wave_file(filename)
                        play_obj = wave_obj.play()
                        play_obj.wait_done()

                        time.sleep(0.2)

                        # Attempt to delete the audio file (with retries)
                        for _ in range(5):
                            try:
                                if os.path.exists(filename):
                                    os.remove(filename)
                                    print(f"Deleted file: {filename}")
                                break
                            except PermissionError as e:
                                if e.errno == errno.EACCES:
                                    print(f"Still locked: {filename}, retrying...")
                                    time.sleep(0.3)
                                else:
                                    print(f"Delete error: {e}")
                                    break

                        last_alert_time_by_label[label] = now

                    except Exception as e:
                        print(f"Error during playback or cleanup: {e}")
                else:
                    print(f"TTS API failed for: {label}")

    cv2.imshow("YOLO Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup: release webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
