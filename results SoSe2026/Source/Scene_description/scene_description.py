import cv2
import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Loading scene description model...")

        # Using more efficient BLIP model instead of GIT
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print("Scene description model loaded!")

    def get_scene_description(self, frame):
        try:
            # Convert frame from BGR to RGB and to PIL Image
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)

            # Process the image
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate caption
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=1.0,
                repetition_penalty=1.5
            )

            # Decode the caption
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'

            return caption

        except Exception as e:
            print(f"Error in scene description: {str(e)}")
            return "Unable to describe the scene."

def main():
    # Initialize the scene describer
    try:
        describer = SceneDescriber()
    except Exception as e:
        print(f"Error initializing the model: {str(e)}")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera")
        return

    print("\nPress 'd' to get scene description")
    print("Press 'q' to quit")
    print("\nCamera is now active...")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break

        # Display the frame
        cv2.imshow('Scene Description', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'd' is pressed, get scene description
        if key == ord('d'):
            description = describer.get_scene_description(frame)
            print("\nScene Description:", description)

        # If 'q' is pressed, quit
        elif key == ord('q'):
            break

    # Release everything when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
