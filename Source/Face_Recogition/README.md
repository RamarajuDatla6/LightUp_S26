# Real-Time Face Recognition System (Dlib + OpenCV)

This Python script implements a real-time face recognition system using OpenCV and Dlib. It detects faces from a webcam, matches them with known faces, and displays names with visual overlays.

---

## 📂 Project Structure

```
project/
│
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── known_faces/
│   ├── person1.jpg
│   ├── person2.png
│   └── ...
│
├── face_3.py
├── requirements.txt
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Clone the repository (or place the script and folders as shown above).

### 2. Install dependencies:
```bash
pip install -r requirements.txt
```

If `dlib` fails to install:
- On Ubuntu:
    ```bash
    sudo apt-get install build-essential cmake
    sudo apt-get install libgtk-3-dev
    sudo apt-get install libboost-all-dev
    pip install dlib
    ```
- On Windows:  
  Install a precompiled binary or build using Visual Studio. Refer to: https://pypi.org/project/dlib/

---

### 3. Prepare model files

Download and place the following pretrained models in the `models/` folder:

- [`shape_predictor_68_face_landmarks.dat`]
- [`dlib_face_recognition_resnet_model_v1.dat`]

Extract them and place them in `models/`.

---

### 4. Add known faces

Place images of known people in the `known_faces/` directory.  
Each image's filename (without extension) will be used as the person's name.

Example:
```
known_faces/
├── alice.jpg   -> Name: alice
├── bob.png     -> Name: bob
```

---

## ▶️ Running the Script

```bash
python face_3.py
```

- The webcam will open and start detecting faces.
- Recognized faces will be labeled.
- Press `q` to quit.

---

## 🧠 Features

- Real-time face detection using `dlib.get_frontal_face_detector()`
- 68-point facial landmarks detection
- Face embeddings using Dlib's ResNet model
- Threaded architecture for efficient frame capture and processing
- Visual overlays with names and corner framing
- FPS counter

---

## 📌 Notes

- You can adjust the face recognition threshold (`0.6`) in the code to tune sensitivity.
- Works best in good lighting and with frontal face images.
- Add more diverse known face images for improved recognition accuracy.

---


