import cv2
import numpy as np
from PIL import Image
import dlib
import os
from ultralytics import YOLO
import torch
import torch.backends.cudnn as cudnn
from transformers import BlipProcessor, BlipForConditionalGeneration
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    cudnn.benchmark = True

def load_models():
    """Load all required models"""
    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        models_dir = os.path.join(project_root, 'models')
        
        print(f"Loading models from: {models_dir}")
        
        object_model = YOLO("yolov8n.pt")
        face_net = cv2.dnn.readNet(
            os.path.join(models_dir, 'deploy.prototxt'),
            os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        )
        shape_predictor = dlib.shape_predictor(
            os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
        )
        face_recognition_model = dlib.face_recognition_model_v1(
            os.path.join(models_dir, 'dlib_face_recognition_resnet_model_v1.dat')
        )
        return object_model, face_net, shape_predictor, face_recognition_model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        return None, None, None, None

def load_known_faces(face_net, shape_predictor, face_recognition_model):
    """Load known faces for face recognition"""

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    known_faces_directory = os.path.join(project_root, 'known_faces')
    
    print(f"Loading known faces from: {known_faces_directory}")
    
    known_face_encodings = []
    known_face_names = []

    if not os.path.exists(known_faces_directory):
        print("No known_faces directory found")
        return known_face_encodings, known_face_names

    for file_name in os.listdir(known_faces_directory):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            try:
                image_path = os.path.join(known_faces_directory, file_name)
                image = cv2.imread(image_path)
                if image is None:
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                face_net.setInput(blob)
                detections = face_net.forward()

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        rect = dlib.rectangle(x1, y1, x2, y2)
                        shape = shape_predictor(rgb_image, rect)
                        encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_image, shape))
                        known_face_encodings.append(encoding)
                        known_face_names.append(os.path.splitext(file_name)[0])
                        break
            except Exception:
                continue

    return known_face_encodings, known_face_names

def process_frames(frame, face_net, shape_predictor, face_recognition_model, known_face_encodings, known_face_names):
    """Process face recognition on a frame"""
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward()

    result_frame = frame.copy()
    current_names = set()
    unknown_count = 0  # Track number of unknown faces
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 - x1 > 0 and y2 - y1 > 0:
                rect = dlib.rectangle(x1, y1, x2, y2)
                shape = shape_predictor(rgb_frame, rect)
                face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_frame, shape))

                name = "Unknown"
                if len(known_face_encodings) > 0:
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    min_distance = np.min(distances)
                    if min_distance < 0.55:
                        index = np.argmin(distances)
                        name = known_face_names[index]
                        current_names.add(name)
                    else:
                        unknown_count += 1  # Increment unknown count


                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(result_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add unknown faces to result if any were detected
    if unknown_count > 0:
        current_names.add(f"{unknown_count} unknown face{'s' if unknown_count > 1 else ''}")

    return result_frame, current_names

class SceneDescriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading scene description model... This might take a few seconds...")


        from transformers import BlipProcessor, BlipForConditionalGeneration

        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device)


        if torch.cuda.is_available():
            self.model = torch.compile(self.model)
        self.model.eval()  # Set to evaluation mode
        print("Scene description model loaded successfully!")

        # Initialize cache for recent descriptions
        self.cache = {}
        self.cache_size = 5
        self.last_process_time = 0
        self.process_interval = 1.0  # Process every 1 second

    def get_frame_hash(self, frame):
        """Generate a simple hash for the frame to use as cache key"""

        small_frame = cv2.resize(frame, (32, 32))
        return hash(small_frame.tobytes())

    @torch.no_grad()
    def get_scene_description(self, frame):
        current_time = time.time()


        if current_time - self.last_process_time < self.process_interval:

            if self.cache:
                return list(self.cache.values())[-1]
            return "Processing scene..."


        frame_hash = self.get_frame_hash(frame)


        if frame_hash in self.cache:
            return self.cache[frame_hash]

        try:

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)


            inputs = self.processor(images=image, return_tensors="pt").to(self.device)


            outputs = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=3,
                length_penalty=1.0,
                num_return_sequences=1,
                temperature=0.7,
            )


            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)


            self.cache[frame_hash] = generated_text
            if len(self.cache) > self.cache_size:
                self.cache.pop(next(iter(self.cache)))

            self.last_process_time = current_time
            return generated_text

        except Exception as e:
            print(f"Error in scene description: {str(e)}")
            return "Unable to describe the scene." 