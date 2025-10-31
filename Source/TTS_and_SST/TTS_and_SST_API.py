from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import sounddevice as sd
import queue
import json
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz
import os
import requests
import zipfile
from pydantic import BaseModel
from typing import Optional, List
import wave
import tempfile
import shutil
from TTS.api import TTS
import numpy as np
import io
import soundfile as sf
import time

app = FastAPI(title="Combined Speech API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define keyword-command mapping
with open("commands.json", "r") as f:
    KEYWORDS = json.load(f)

MATCH_THRESHOLD = 85  # Tune this (70–95) for leniency vs. precision


# Initialize models
def initialize_models():
    # Initialize VOSK model for speech recognition
    model_path = "models/speech/model/vosk-model-small-en-us-0.15"
    if not os.path.exists(model_path):
        ensure_vosk_model()
    vosk_model = Model(model_path)

    # Initialize TTS model
    tts_model = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)

    return vosk_model, tts_model


def ensure_vosk_model():
    if not os.path.exists("model"):
        os.makedirs("model")

    model_url = "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip"
    model_zip_path = "model/vosk-model-small-en-us-0.15.zip"

    if not os.path.exists("models/speech/model/vosk-model-small-en-us-0.15"):
        print("Downloading VOSK model (may take 1-2 minutes)...")
        with requests.get(model_url, stream=True) as r:
            with open(model_zip_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        with zipfile.ZipFile(model_zip_path, 'r') as zip_ref:
            zip_ref.extractall("model")


# Initialize models at startup
vosk_model, tts_model = initialize_models()


class SpeechResponse(BaseModel):
    text: str
    matched_command: Optional[str] = None
    confidence_score: Optional[float] = None
    module: Optional[str] = None
    processing_time: Optional[float] = None


class TTSResponse(BaseModel):
    processing_time: float
    audio_duration: float
    real_time_factor: float


@app.get("/")
async def root():
    return {
        "message": "Combined Speech API is running",
        "endpoints": {
            "/recognize": "Convert speech to text",
            "/speak": "Convert text to speech",
            "/audio-devices": "List available audio devices"
        }
    }


@app.post("/recognize", response_model=SpeechResponse)
async def recognize_speech(audio_file: UploadFile = File(...)):
    start_time = time.time()
    temp_file_path = None
    try:
        # Create a temporary file to store the uploaded audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            shutil.copyfileobj(audio_file.file, temp_file)
            temp_file_path = temp_file.name

        # Process the audio file
        wf = wave.open(temp_file_path, "rb")

        # Check if audio file is valid
        if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
            wf.close()
            raise HTTPException(status_code=400, detail="Audio file must be WAV format mono PCM.")

        rec = KaldiRecognizer(vosk_model, wf.getframerate())
        rec.SetWords(True)

        # Read the audio file
        while True:
            data = wf.readframes(4000)
            if len(data) == 0:
                break
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                recognized_text = result.get("text", "").lower()

                best_match = None
                best_score = 0

                for phrase, module in KEYWORDS.items():
                    score = fuzz.partial_ratio(phrase, recognized_text)
                    if score > best_score:
                        best_score = score
                        best_match = (phrase, module)

                wf.close()
                processing_time = time.time() - start_time
                response = {
                    "text": recognized_text,
                    "matched_command": best_match[0] if best_match and best_score >= MATCH_THRESHOLD else None,
                    "confidence_score": best_score if best_match else None,
                    "module": best_match[1] if best_match and best_score >= MATCH_THRESHOLD else None,
                    "processing_time": processing_time
                }
                return response

        # Get final result
        result = json.loads(rec.FinalResult())
        recognized_text = result.get("text", "").lower()

        best_match = None
        best_score = 0

        for phrase, module in KEYWORDS.items():
            score = fuzz.partial_ratio(phrase, recognized_text)
            if score > best_score:
                best_score = score
                best_match = (phrase, module)

        wf.close()
        processing_time = time.time() - start_time
        response = {
            "text": recognized_text,
            "matched_command": best_match[0] if best_match and best_score >= MATCH_THRESHOLD else None,
            "confidence_score": best_score if best_match else None,
            "module": best_match[1] if best_match and best_score >= MATCH_THRESHOLD else None,
            "processing_time": processing_time
        }
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception as e:
                print(f"Warning: Could not delete temporary file: {e}")


@app.post("/speak")
async def speak(text: str = Form(...)):
    start_time = time.time()
    try:
        # Generate waveform
        wav = tts_model.tts(text)
        sr = tts_model.synthesizer.output_sample_rate

        # Calculate audio duration (approximate)
        audio_duration = len(text.split()) * 0.3  # Rough estimate: 0.3 seconds per word
        processing_time = time.time() - start_time
        rtf = audio_duration / processing_time

        # Save to buffer in-memory
        buffer = io.BytesIO()
        sf.write(buffer, wav, sr, format='WAV')
        buffer.seek(0)

        # Return audio stream with metadata
        return StreamingResponse(
            buffer,
            media_type="audio/wav",
            headers={
                "X-Processing-Time": str(processing_time),
                "X-Audio-Duration": str(audio_duration),
                "X-Real-Time-Factor": str(rtf)
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio-devices")
async def list_audio_devices():
    try:
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        return {
            "input_devices": input_devices,
            "output_devices": output_devices
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000) 