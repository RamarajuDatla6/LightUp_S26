# server.py

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
import cv2
import numpy as np
import base64
from pydantic import BaseModel, Field
from typing import Optional, List, Union, Literal, Dict, Any
import uvicorn
import time

from models import load_models, load_known_faces, process_frames, SceneDescriber
from speech import speak_async

# Power simulation
from power_sim import list_models as power_list_models
from power_sim import estimate_from_timings as power_estimate_from_timings
from power_sim import estimate_from_workload as power_estimate_from_workload
from isaac_vlm import IsaacVLM


app = FastAPI(title="Vision Assistant Server", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"\n VALIDATION ERROR (422) ")
    print(f"Request URL: {request.url}")
    print(f"Request method: {request.method}")
    print(f"Request headers: {dict(request.headers)}")
    try:
        body = await request.body()
        print(f"Request body length: {len(body)}")
        if len(body) < 1000:
            print(f"Request body: {body.decode('utf-8')}")
        else:
            print(f"Request body (first 500 chars): {body[:500].decode('utf-8')}...")
    except Exception as e:
        print(f"Could not read request body: {e}")

    print(f"Validation errors: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={
            "detail": "Request validation failed",
            "errors": exc.errors(),
            "debug_info": {
                "url": str(request.url),
                "method": request.method,
                "error_count": len(exc.errors()),
            },
        },
    )

# Global models
object_model = None
face_net = None
shape_predictor = None
face_recognition_model = None
known_face_encodings = []
known_face_names = []
scene_describer = None
isaac_client = None



# class FrameRequest(BaseModel):
#     frame_data: str = Field(..., description="base64 encoded image data")
#     mode: Literal["face", "object", "scene"] = Field(..., description="Processing mode")

#     # power sim controls (client can override)
#     model_id: Optional[str] = Field(
#         default="snapdragon_8_gen_3",
#         description="SoC model id for power simulation",
#     )
#     policy: Optional[Literal["balanced", "performance", "battery"]] = Field(
#         default="balanced",
#         description="Power policy for simulation",
#     )

class FrameRequest(BaseModel):
    frame_data: str = Field(..., description="base64 encoded image data")
    mode: Literal["face", "object", "scene", "isaac"] = Field(..., description="Processing mode")

    # power sim controls (client can override)
    model_id: Optional[str] = Field(default="snapdragon_8_gen_3")
    policy: Optional[Literal["balanced", "performance", "battery"]] = Field(default="balanced")

    # Isaac VLM controls
    vlm_prompt: Optional[str] = Field(
        default="Describe the scene for a blind user in 1 short sentence.",
        description="Prompt for Isaac VLM mode",
    )

    # vlm_hint: Optional[Literal["NONE", "BOX", "POINT", "POLYGON"]] = Field(
    #     default="NONE",
    #     description="Isaac hint mode",
    # )




class ProcessingResponse(BaseModel):
    detection_result: Optional[Union[List[str], str]] = None
    success: bool = True
    error: Optional[str] = None

    # Added: measured timings + power estimate
    timings_ms: Optional[Dict[str, float]] = None
    power: Optional[Dict[str, Any]] = None


class PowerEstimateRequest(BaseModel):
    model_id: str
    mode: Literal["face", "object", "scene"]
    policy: Optional[Literal["balanced", "performance", "battery"]] = "balanced"
    timings_ms: Dict[str, float]
    include_overhead: bool = True
    overhead_mw: float = 250.0


class WorkloadEstimateRequest(BaseModel):
    model_id: str
    fps: float = Field(30.0, ge=0.1, le=240.0)
    util_cpu: float = Field(0.4, ge=0.0, le=1.0)
    util_gpu: float = Field(0.0, ge=0.0, le=1.0)
    util_npu: float = Field(0.0, ge=0.0, le=1.0)
    policy: Optional[Literal["balanced", "performance", "battery"]] = "balanced"
    battery_wh: Optional[float] = Field(None, description="Override battery size (Wh)")


class SpeechRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup_event():
    global object_model, face_net, shape_predictor, face_recognition_model
    global known_face_encodings, known_face_names, scene_describer
    # global isaac_client
    # isaac_client = IsaacVLMClient(
    #     model="isaac-0.2-2b-preview",
    #     process_interval_s=1.5,
    #     cache_size=8,
    # )


    print("Loading models...")
    try:
        object_model, face_net, shape_predictor, face_recognition_model = load_models()
        global isaac_vlm
        print("Model loading results:")
        print(f"  object_model: {'✓' if object_model is not None else '✗'}")
        print(f"  face_net: {'✓' if face_net is not None else '✗'}")
        print(f"  shape_predictor: {'✓' if shape_predictor is not None else '✗'}")
        print(f"  face_recognition_model: {'✓' if face_recognition_model is not None else '✗'}")

        if all(m is not None for m in [object_model, face_net, shape_predictor, face_recognition_model]):
            known_face_encodings, known_face_names = load_known_faces(
                face_net, shape_predictor, face_recognition_model
            )
            print(f"Loaded {len(known_face_names)} known faces")

            scene_describer = SceneDescriber()
            print("All models loaded successfully!")
        else:
            print("Failed to load some models!")
        #     global isaac_vlm
        # try:
        #     print("Loading Isaac VLM...")
        #     isaac_vlm = IsaacVLM()
        #     print("Isaac VLM loaded successfully!")
        # except Exception as e:
        #     isaac_vlm = None
        #     print(f"Isaac VLM NOT loaded: {e}")
        try:
            print("Loading Isaac VLM...")
            isaac_vlm = IsaacVLM()
            print("Isaac VLM loaded successfully!")
        except Exception as e:
            isaac_vlm = None
            print(f"Isaac VLM NOT loaded: {e}")


    except Exception as e:
        print(f"Error during model loading: {e}")
        import traceback
        traceback.print_exc()


def decode_frame(frame_data: str) -> Optional[np.ndarray]:
    try:
        if "," in frame_data:
            frame_data = frame_data.split(",")[1]

        img_bytes = base64.b64decode(frame_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return frame
    except Exception as e:
        print(f"Error decoding frame: {e}")
        return None


@app.post("/process_frame", response_model=ProcessingResponse)
async def process_frame_endpoint(request: FrameRequest):
    t0_total = time.perf_counter()
    try:
        # if object_model is None or face_net is None or scene_describer is None:
        #     raise HTTPException(status_code=500, detail="Models not loaded properly")
        # --- mode-aware model guard ---
        if request.mode == "isaac":
            if isaac_vlm is None:
                raise HTTPException(status_code=500, detail="Isaac VLM not loaded properly")
        else:
            if object_model is None or face_net is None or scene_describer is None:
                raise HTTPException(status_code=500, detail="Models not loaded properly")

        # --- decode timing ---
        t0 = time.perf_counter()
        frame = decode_frame(request.frame_data)
        t_decode = (time.perf_counter() - t0) * 1000.0
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid frame data")

        detection_result = None
        t_infer = 0.0
        t_post = 0.0

        if request.mode == "face":
            t0 = time.perf_counter()
            _, current_names = process_frames(
                frame, face_net, shape_predictor, face_recognition_model,
                known_face_encodings, known_face_names
            )
            t_infer = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            if current_names:
                detection_result = list(current_names)
            t_post = (time.perf_counter() - t0) * 1000.0

        elif request.mode == "object":
            t0 = time.perf_counter()
            results = object_model(frame)
            t_infer = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            object_names = set()
            if results and results[0].boxes is not None:
                for box in results[0].boxes:
                    if float(box.conf[0]) > 0.5:
                        label = results[0].names[int(box.cls[0])]
                        object_names.add(label)
            if object_names:
                detection_result = list(object_names)
            t_post = (time.perf_counter() - t0) * 1000.0

        elif request.mode == "scene":
            t0 = time.perf_counter()
            description = scene_describer.get_scene_description(frame)
            t_infer = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            detection_result = description
            t_post = (time.perf_counter() - t0) * 1000.0
        
        elif request.mode == "isaac":
            if isaac_vlm is None:
                raise HTTPException(status_code=500, detail="Isaac VLM not loaded")

            t0 = time.perf_counter()
            text = isaac_vlm.describe(
            frame,
            prompt=request.vlm_prompt or
            "Describe the scene for a blind user in 1 short sentence.",
        )

            t_infer = (time.perf_counter() - t0) * 1000.0

            t0 = time.perf_counter()
            detection_result = text
            t_post = (time.perf_counter() - t0) * 1000.0





        else:
            raise HTTPException(status_code=400, detail="Invalid mode")

        t_total = (time.perf_counter() - t0_total) * 1000.0

        timings_ms = {
            "decode": round(t_decode, 3),
            "inference": round(t_infer, 3),
            "post": round(t_post, 3),
            "total": round(t_total, 3)
        }

        power_dict = None
        try:
            est = power_estimate_from_timings(
                model_id=request.model_id or "snapdragon_8_gen_3",
                timings_ms={"decode": timings_ms["decode"], "inference": timings_ms["inference"], "post": timings_ms["post"]},
                # mode=request.mode,
                mode=("scene" if request.mode == "isaac" else request.mode),

                policy=request.policy or "balanced",
                extra={
                    "include_overhead": True,
                    "overhead_mw": 250.0,
                    "include_memory": True
                }
            )
            power_dict = est.to_dict()
        except Exception as pe:
            power_dict = {"error": str(pe)}

        return ProcessingResponse(
            detection_result=detection_result,
            success=True,
            timings_ms=timings_ms,
            power=power_dict
        )

    except HTTPException:
        raise
    except Exception as e:
        return ProcessingResponse(success=False, error=str(e))


@app.post("/speak")
async def speak_endpoint(request: SpeechRequest):
    try:
        speak_async(request.text)
        return {"success": True, "message": "Speech started"}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "models_loaded": {
            "object_model": object_model is not None,
            "face_net": face_net is not None,
            "shape_predictor": shape_predictor is not None,
            "face_recognition_model": face_recognition_model is not None,
            "scene_describer": scene_describer is not None,
            "isaac_vlm_loaded": isaac_vlm is not None,

        },
        "known_faces": len(known_face_names),
        # helpful for client default UI
        "battery_wh": 19.0,
    }


@app.get("/power/models")
async def power_models():
    try:
        return {"models": power_list_models()}
    except Exception as e:
        return {
            "models": [],
            "error": str(e),
            "hint": "Check power_sim/profiles.json exists and contains valid JSON."
        }



@app.post("/power/estimate")
async def power_estimate(req: PowerEstimateRequest):
    est = power_estimate_from_timings(
        model_id=req.model_id,
        timings_ms=req.timings_ms,
        mode=req.mode,
        policy=req.policy,
        extra={"include_overhead": req.include_overhead, "overhead_mw": req.overhead_mw},
    )
    return est.to_dict()


@app.post("/power/workload")
async def power_workload(req: WorkloadEstimateRequest):
    est = power_estimate_from_workload(
        model_id=req.model_id,
        fps=req.fps,
        util_cpu=req.util_cpu,
        util_gpu=req.util_gpu,
        util_npu=req.util_npu,
        policy=req.policy,
        battery_wh=req.battery_wh,
    )
    return est.to_dict()


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
