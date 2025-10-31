# Scene Description using BLIP

This project uses the BLIP (Bootstrapping Language-Image Pretraining) model from Hugging Face to describe scenes captured from a webcam in real-time.

## Features

- Captures video using OpenCV
- Describes the current frame using the BLIP image captioning model
- Displays the camera feed with live interaction via keyboard

## How It Works

1. Captures frames from the default webcam.
2. When you press **'d'**, the model generates a description of the current frame.
3. Press **'q'** to quit the application.

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

## Usage

Run the script with:

```bash
python scene_description.py
```

Ensure your system has a webcam connected and functional.

## Notes

- Uses GPU if available; otherwise falls back to CPU.
- Based on the [`Salesforce/blip-image-captioning-base`](https://huggingface.co/Salesforce/blip-image-captioning-base) model.

## License

MIT License
