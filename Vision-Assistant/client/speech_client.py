import os
import queue
import json
import threading
import sounddevice as sd
from vosk import Model, KaldiRecognizer
from rapidfuzz import fuzz

# Load commands.json
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
commands_path = os.path.join(project_root, "commands.json")
with open(commands_path, "r") as f:
    VOICE_COMMANDS = json.load(f)

_vosk_model = None
_audio_stream = None
_recognizer = None

def initialize_speech_recognition():
    """Initialize speech recognition components"""
    global _vosk_model, _audio_stream, _recognizer

    try:
        if _vosk_model is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            model_path = os.path.join(project_root, "models", "speech", "model", "vosk-model-small-en-us-0.15")
            print(f"Loading Vosk model from: {model_path}")
            _vosk_model = Model(model_path)
            _recognizer = KaldiRecognizer(_vosk_model, 16000)

        audio_queue = queue.Queue()

        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            try:
                audio_queue.put_nowait(bytes(indata))
            except Exception as e:
                print(f"Callback error: {e}")

        if _audio_stream is None:
            print("Creating audio stream...")
            _audio_stream = sd.RawInputStream(
                samplerate=16000,
                blocksize=8000,
                channels=1,
                dtype='int16',
                callback=audio_callback
            )

        return _recognizer, audio_queue, audio_callback, _audio_stream

    except Exception as e:
        print(f"Speech recognition initialization error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

def process_speech(recognizer, audio_queue, stop_event, mode_callback):
    """Process speech in a separate thread"""
    print("Speech recognition thread started")
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.1)
            if audio_data and recognizer.AcceptWaveform(audio_data):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").lower().strip()
                if text:
                    print(f"Recognized: '{text}'")
                    for command, action in VOICE_COMMANDS.items():
                        ratio = fuzz.ratio(text, command)
                        print(f"Matching '{text}' with '{command}': {ratio}%")
                        if ratio > 80:
                            print(f"Command matched: {command}")
                            mode_callback(action)
                            break
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Speech processing error: {e}")

def cleanup_speech_recognition():
    """Clean up speech recognition resources"""
    global _audio_stream
    if _audio_stream is not None:
        try:
            _audio_stream.stop()
            _audio_stream.close()
        except Exception as e:
            print(f"Error cleaning up audio stream: {e}")
        _audio_stream = None 