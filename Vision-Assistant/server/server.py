from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pydantic import BaseModel, ValidationError, Field
from typing import Optional, List, Union, Literal
import uvicorn
import json

from models import load_models, load_known_faces, process_frames, SceneDescriber
from speech import speak_async

# Initialize FastAPI app
app = FastAPI(title="Vision Assistant Server", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Custom exception handler for validation errors (422)
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n VALIDATION ERROR (422) ")
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    
    # Try to get request body
    try:
        body = await request.body()
        print(f"Request body length: {len(body)}")
        if len(body) < 1000:  # Only print small bodies
            print(f"Request body: {body.decode('utf-8')}")
        else:
            print(f"Request body (first 500 chars): {body[:500].decode('utf-8')}...")
    except Exception as e:
        print(f"Could not read request body: {e}")
    
    print(f"Validation errors: {exc.errors()}")
    print(f"Exception details: {exc}")
    
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "error_count": len(exc.errors())
            }
        }
    )

# Global variables to store models
object_model = None
face_net = None
shape_predictor = None
face_recognition_model = None
known_face_encodings = []
known_face_names = []
scene_describer = None

class FrameRequest(BaseModel):
    frame_data: str = Field(..., description="base64 encoded image data")
    mode: Literal['face', 'object', 'scene'] = Field(..., description="Processing mode: face, object, or scene")

class ProcessingResponse(BaseModel):
    detection_result: Optional[Union[List[str], str]] = None
    success: bool = True
    error: Optional[str] = None

class SpeechRequest(BaseModel):
    text: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on server startup"""
    global object_model, face_net, shape_predictor, face_recognition_model
    global known_face_encodings, known_face_names, scene_describer
    
    print("Loading models...")
    
    try:
        # Load models
        object_model, face_net, shape_predictor, face_recognition_model = load_models()
        
        print(f"Model loading results:")
        print(f"  object_model: {'✓' if object_model is not None else '✗'}")
        print(f"  face_net: {'✓' if face_net is not None else '✗'}")
        print(f"  shape_predictor: {'✓' if shape_predictor is not None else '✗'}")
        print(f"  face_recognition_model: {'✓' if face_recognition_model is not None else '✗'}")
        
        if all(model is not None for model in [object_model, face_net, shape_predictor, face_recognition_model]):
            # Load known faces
            known_face_encodings, known_face_names = load_known_faces(
                face_net, shape_predictor, face_recognition_model
            )
            print(f"Loaded {len(known_face_names)} known faces")
            
            # Initialize scene describer
            scene_describer = SceneDescriber()
            
            print("All models loaded successfully!")
        else:
            print("Failed to load some models!")
            
    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()

def decode_frame(frame_data: str) -> np.ndarray:
    """Decode base64 frame data to numpy array"""
    try:

        if ',' in frame_data:
            frame_data = frame_data.split(',')[1]

        img_bytes = base64.b64decode(frame_data)

        nparr = np.frombuffer(img_bytes, np.uint8)

        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return frame
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None



@app.post("/process_frame", response_model=ProcessingResponse)
async def process_frame_endpoint(request: FrameRequest):
    """Process a frame based on the specified mode - returns only detection results, no visual output"""
    try:
        print(f"\n=== PROCESSING REQUEST ===")
        print(f"Mode: {request.mode}")
        print(f"Frame data length: {len(request.frame_data) if request.frame_data else 'None'}")
        
        # Check if models are loaded
        if object_model is None or face_net is None or scene_describer is None:
            print("Models not loaded properly")
            raise HTTPException(status_code=500, detail="Models not loaded properly")
        
        print(" All models are loaded")
        
        # Decode the frame
        frame = decode_frame(request.frame_data)
        if frame is None:
            print(" Failed to decode frame data")
            raise HTTPException(status_code=400, detail="Invalid frame data")
        
        print(f" Frame decoded successfully, shape: {frame.shape}")
        
        detection_result = None
        

        if request.mode == 'face':
            print("Processing face recognition...")

            _, current_names = process_frames(
                frame, face_net, shape_predictor, face_recognition_model,
                known_face_encodings, known_face_names
            )
            if current_names:
                detection_result = list(current_names)
                print(f"Face detection result: {detection_result}")
            
        elif request.mode == 'object':
            print("Processing object detection...")
            results = object_model(frame)
            
            object_names = set()
            if results[0].boxes is not None:
                for box in results[0].boxes:
                    if float(box.conf[0]) > 0.5:  # Confidence threshold
                        label = results[0].names[int(box.cls[0])]
                        object_names.add(label)
            
            if object_names:
                detection_result = list(object_names)
                print(f"Object detection result: {detection_result}")
            
        elif request.mode == 'scene':
            print("Processing scene description...")
            description = scene_describer.get_scene_description(frame)
            detection_result = description
            print(f"Scene description result: {detection_result}")
            
        else:
            raise HTTPException(status_code=400, detail="Invalid mode")
        
        print("Detection processing completed successfully")
        
        return ProcessingResponse(
            detection_result=detection_result,
            success=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()
        return ProcessingResponse(
            detection_result=None,
            success=False,
            error=str(e)
        )

@app.post("/speak")
async def speak_endpoint(request: SpeechRequest):
    """Text-to-speech endpoint"""
    try:
        speak_async(request.text)
        return {"success": True, "message": "Speech started"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/test_request")
async def test_request(request: FrameRequest):
    """Test endpoint to debug request format"""
    try:
        print(f"Test request received - mode: {request.mode}")
        print(f"Frame data length: {len(request.frame_data)}")
        
        #  decode frame
        frame = decode_frame(request.frame_data)
        if frame is not None:
            print(f"Frame decoded successfully: shape {frame.shape}")
            return {"success": True, "message": "Request format is valid"}
        else:
            return {"success": False, "error": "Failed to decode frame"}
            
    except Exception as e:
        print(f"Test request error: {e}")
        return {"success": False, "error": str(e)}

@app.get("/debug/schema")
async def get_request_schema():
    """Debug endpoint to show expected request schema"""
    return {
        "FrameRequest_schema": FrameRequest.schema(),
        "ProcessingResponse_schema": ProcessingResponse.schema(),
        "SpeechRequest_schema": SpeechRequest.schema(),
        "example_request": {
            "frame_data": "base64_encoded_image_data_here",
            "mode": "face"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "object_model": object_model is not None,
            "face_net": face_net is not None,
            "shape_predictor": shape_predictor is not None,
            "face_recognition_model": face_recognition_model is not None,
            "scene_describer": scene_describer is not None
        },
        "known_faces": len(known_face_names)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 