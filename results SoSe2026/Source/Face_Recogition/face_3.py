"""
face_3.py

Core simulation framework for Light UP.

Provides tools for configuring, running, and analyzing simulations.

Author: Nishant Hiteshbhai Kachhadiya <n.kachhadiya@oth-aw.de> 
License: CC BY 1.0 
Creation Date: 2025-06-15
"""




import cv2
import dlib
import numpy as np
import os
import time
import threading
import queue

# Initialize webcam
video_capture = cv2.VideoCapture(0)


model_directory = 'models'
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor(os.path.join(model_directory, 'shape_predictor_68_face_landmarks.dat'))
face_recognition_model = dlib.face_recognition_model_v1(os.path.join(model_directory, 'dlib_face_recognition_resnet_model_v1.dat'))

# Load known faces
known_face_encodings = []
known_face_names = []

known_faces_directory = 'known_faces'
for file_name in os.listdir(known_faces_directory):
    if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        image_path = os.path.join(known_faces_directory, file_name)
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = face_detector(rgb_image)
        if detections:
            shape = shape_predictor(rgb_image, detections[0])
            encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_image, shape))
            known_face_encodings.append(encoding)
            known_face_names.append(os.path.splitext(file_name)[0])


frame_queue = queue.Queue(maxsize=10)
processed_frame = None
frame_lock = threading.Lock()

# Thread to capture frames
def capture_frames():
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)

# Thread to process frames
def process_frames():
    global processed_frame
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()

            # Resize
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            face_names = []
            faces = face_detector(rgb_small_frame)

            for face in faces:
                shape = shape_predictor(rgb_small_frame, face)
                face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_small_frame, shape))

                name = "Unknown"
                if known_face_encodings:
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    min_distance = np.min(distances)
                    if min_distance < 0.6:
                        index = np.argmin(distances)
                        name = known_face_names[index]

                face_names.append((face, name))



            for face, name in face_names:

                left = face.left() * 2
                top = face.top() * 2
                right = face.right() * 2
                bottom = face.bottom() * 2


                corner_length = 60

                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)


                cv2.line(frame, (left, top), (left + corner_length, top), color, 5)
                cv2.line(frame, (left, top), (left, top + corner_length), color, 5)


                cv2.line(frame, (right, top), (right - corner_length, top), color, 5)
                cv2.line(frame, (right, top), (right, top + corner_length), color, 5)


                cv2.line(frame, (left, bottom), (left + corner_length, bottom), color, 5)
                cv2.line(frame, (left, bottom), (left, bottom - corner_length), color, 5)


                cv2.line(frame, (right, bottom), (right - corner_length, bottom), color, 5)
                cv2.line(frame, (right, bottom), (right, bottom - corner_length), color, 5)


                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 4)


            with frame_lock:
                processed_frame = frame


capture_thread = threading.Thread(target=capture_frames, daemon=True)
process_thread = threading.Thread(target=process_frames, daemon=True)
capture_thread.start()
process_thread.start()


fps_start_time = time.time()
while True:
    with frame_lock:
        if processed_frame is not None:

            fps_end_time = time.time()
            fps = int(1 / (fps_end_time - fps_start_time))
            fps_start_time = fps_end_time
            cv2.putText(processed_frame, f'FPS: {fps}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.imshow("Face Recognition", processed_frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()