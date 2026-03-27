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
        #This model finds faces inside an image.
        face_net = cv2.dnn.readNet(
            os.path.join(models_dir, 'deploy.prototxt'),
            os.path.join(models_dir, 'res10_300x300_ssd_iter_140000.caffemodel')
        )

        #This loads a model that detects 68 face landmarks like:eyes,nose, mouth,jawline
        #This is useful for alignment, cropping the face, etc.
        shape_predictor = dlib.shape_predictor(
            os.path.join(models_dir, 'shape_predictor_68_face_landmarks.dat')
        )

        #This loads the face embedding model—a network that converts a face into a 128-dimensional vector to identify who the person is.
        #This model does actual face recognition (who is the person?), not just detection.
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
                image = cv2.imread(image_path) #read the image and convert to BGR format
                if image is None:
                    continue

                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #convert the image from BGR to RGB
                h, w = image.shape[:2] #takes first 2 values:only height and width and removes the BGR
                blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
                face_net.setInput(blob) #Process the image
                detections = face_net.forward() #run the faces and tells which face was found

                #The number of face detections the model produced:shape[2]
                for i in range(detections.shape[2]): 
                    confidence = detections[0, 0, i, 2] #out of detected faces find the confidence score
                    if confidence > 0.5:
                        #3:7 -> x1,y1 ,x2, y2 and multiply it by width and height of frame to get actual values
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        #rectangle on the face x1=left corner of face x2 = right y1 top y2 bottom
                        rect = dlib.rectangle(x1, y1, x2, y2)  
                        #Predict 68 points on the face
                        shape = shape_predictor(rgb_image, rect) 
                        #Convert the face into a unique 128-number fingerprint
                        encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_image, shape))
                        #Put the face’s 128-number code into a list.
                        known_face_encodings.append(encoding)
                        #Save the person's name (from the file name) sai.png to sai
                        known_face_names.append(os.path.splitext(file_name)[0])
                        break  #Stop after first face in this image
            except Exception:
                continue

    return known_face_encodings, known_face_names

def process_frames(frame, face_net, shape_predictor, face_recognition_model, known_face_encodings, known_face_names):
    """Process face recognition on a frame"""
    h, w = frame.shape[:2]
    #resize the frame to 300x300
    #False False mean: Don’t flip the picture, Don’t swap the colour channels.So the picture stays normal.
    #1.0 means dont change the pixel value
    #BlobFromImage: Convert the photo into a special format that the model can understand.
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
    face_net.setInput(blob)
    detections = face_net.forward() #run the faces and tells which face was found

    #Make a copy of the frame where you will draw rectangles and names.
    #Original frame stays untouched.
    result_frame = frame.copy()

    #A set to store names of people seen in this frame.
    # Set is used so each name appears only once, even if same person appears twice.
    current_names = set()
    unknown_count = 0  # Track number of unknown faces

    # #Convert from BGR → RGB because dlib/face recognition expects RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #The number of face detections the model produced:shape[2]
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2] #2 holds the confidence value
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int) #Convert from float → integer (pixels).
            #Make sure the box doesn’t go outside the image.
            #Clamp the values so they are between 0 and width/height.
            #If x1 is less than 0, change it to 0. If x1 is already positive, keep it.
            #If x2 is bigger than image width (w), change it to w. If it is inside the image, keep it.
            #Because the right side cannot go beyond the image width.
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            #Ensure width and height are positive. If not, ignore this detection (maybe it was bad or weird).
            if x2 - x1 > 0 and y2 - y1 > 0:
                rect = dlib.rectangle(x1, y1, x2, y2) #Make a rectangle object for dlib: tells where the face is.
                shape = shape_predictor(rgb_frame, rect) #Find 68 key points on the face: eyes, nose, mouth, jaw.
                #Convert face into a 128-number vector (face fingerprint).
                # This is what you compare with known faces.
                face_encoding = np.array(face_recognition_model.compute_face_descriptor(rgb_frame, shape))

                name = "Unknown" #Assume at first we don’t know this person.
                #Only try matching if we have some known faces stored.
                if len(known_face_encodings) > 0:
                    #Each face is a 128-number vector. 
                    #We subtract the patterns and calculate how different they are.
                    #Smaller distance = more similar
                    #Bigger distance = different faces
                    distances = np.linalg.norm(known_face_encodings - face_encoding, axis=1)
                    min_distance = np.min(distances) #This picks the smallest difference.
                    #This is the magic threshold.
                    #If the distance is smaller than 0.55 → YES this looks like a match
                    #If the distance is bigger → NO, not the same person
                    if min_distance < 0.55: 
                        #argmin gives the POSITION (index) of the smallest value.
                        index = np.argmin(distances) #distances = [0.43, 0.89, 0.22, 0.67] min is at pos 2.. index = 2
                        name = known_face_names[index]
                        current_names.add(name)
                    else:
                        #If not a match, count unknown people
                        unknown_count += 1  # Increment unknown count

                #If the name is not "Unknown" → use green (0,255,0)
                #If the name is "Unknown" → use red (0,0,255)
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(result_frame, (x1, y1), (x2, y2), color, 2) #2 is thickness of line
                #0.6:font size, 2:thickness of text
                cv2.putText(result_frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Add unknown faces to result if any were detected
    if unknown_count > 0:
        current_names.add(f"{unknown_count} unknown face{'s' if unknown_count > 1 else ''}")

    return result_frame, current_names


#The code:
#Loads the BLIP image caption model
#Converts frames into text descriptions
#Uses cache to avoid repeating work
# Only describes once every second
# Returns a sentence like:"A man sitting in a room holding a laptop."
class SceneDescriber:
    def __init__(self):
        #use gpu if available else cpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading scene description model... This might take a few seconds...")

        #Processor = prepares image, Model = generates text
        from transformers import BlipProcessor, BlipForConditionalGeneration
        
        #Load the tool that helps convert images into the format the model likes.
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

        #Load the AI brain that will create the sentence for the image.
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
        )
        self.model.to(self.device) #Put the model either on GPU or CPU.

        #If GPU exists, turbo-boost the model. Make model even faster
        if torch.cuda.is_available():
            self.model = torch.compile(self.model)
        self.model.eval()  # Set to evaluation mode. We are NOT training the model. We are only using it.
        print("Scene description model loaded successfully!")

        # Initialize cache for recent descriptions
        #Cache = memory box, Cache size = only store 5 descriptions
        # Process interval = only describe once every 1 second
        #self.cache looks like this { 874329847234 : "A boy sitting on a chair.",112233445566 : "A table with a laptop."}
        #Hash is the key, Caption is the value
        self.cache = {}
        self.cache_size = 5
        self.last_process_time = 0
        self.process_interval = 1.0  # Process every 1 second

    def get_frame_hash(self, frame):
        """Generate a simple hash for the frame to use as cache key"""
        #Shrinks image to 32×32.Converts to bytes.Hashes it = unique ID.
        #This helps detect if two frames are basically the same.
        small_frame = cv2.resize(frame, (32, 32))
        return hash(small_frame.tobytes())

    @torch.no_grad()
    #Take a camera frame and describe what is happening.
    def get_scene_description(self, frame):
        current_time = time.time()

        #If only 1 second has not passed yet, do not describe again.
        #If old description exists → return it, Else → return “Processing…”
        if current_time - self.last_process_time < self.process_interval:
            if self.cache:
                return list(self.cache.values())[-1]
            return "Processing scene..."
        
        #Create hash of this frame
        frame_hash = self.get_frame_hash(frame)

        #Check if we already described this frame.
        #If I already described this picture, return the saved text.
        if frame_hash in self.cache:
            return self.cache[frame_hash]

        try:
            #Convert image to RGB and turn it into a PIL image.
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)

            #Turn the picture into tensors the model likes.
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)

            #Generate caption
            #max_length=30 → Max 30 words, num_beams=3 → Try 3 guesses to pick best
            #temperature=0.7 → Slight randomness

            outputs = self.model.generate(
                **inputs,
                max_length=30,
                num_beams=3,
                length_penalty=1.0,  #not too short, not too long
                num_return_sequences=1, #show one caption on screen
                temperature=0.7, #Be a little creative, but not too crazy.
            )

            #Turn model output into real text.
            generated_text = self.processor.decode(outputs[0], skip_special_tokens=True)

            #Store the description but keep cache size small.
            self.cache[frame_hash] = generated_text
            if len(self.cache) > self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            #Update last time we processed
            self.last_process_time = current_time
            return generated_text

        except Exception as e:
            print(f"Error in scene description: {str(e)}")
            return "Unable to describe the scene." 