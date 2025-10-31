# Combined TTS and SST API

This is a FastAPI-based web service that provides the following functionalities:
- **Speech-to-Text (STT)**: Converts speech from WAV audio files into text using the VOSK model.
- **Text-to-Speech (TTS)**: Converts text input into spoken audio using the `TTS` library (Tacotron2-DDC).
- **Command Matching**: Detects pre-defined spoken commands using fuzzy matching.
- **Audio Device Listing**: Lists available input/output audio devices using `sounddevice`.

## Features

- 📥 `/recognize`: Upload a WAV file to transcribe and detect commands.
- 📤 `/speak`: Send text and receive TTS audio in WAV format.
- 🎧 `/audio-devices`: Get available input/output audio devices.
- 🌍 CORS enabled for all origins.

## Installation

```bash
git clone <your-repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

## Run the API

```bash
uvicorn TTS_and_SST_API:app --reload
```

## Required Files

- `commands.json`: JSON file containing keyword-command mapping.
- Pretrained models:
  - VOSK: Automatically downloaded on first run.
  - TTS: Tacotron2-DDC via the `TTS` library.

## Example commands.json
```json
{
    "turn on the light": "lights_on",
    "start music": "music_play"
}
```

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Root message and endpoint listing |
| `/recognize` | POST | Upload audio and get recognized text with command |
| `/speak` | POST | Provide text and receive synthesized audio |
| `/audio-devices` | GET | List of available audio devices |

## Notes

- Make sure to use WAV format mono PCM audio files for recognition.
- Internet connection required for downloading models on first use.

---

© 2025 Combined Speech API Project