import cv2
import torch
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        print("Loading scene description model...")

        # Using more efficient BLIP model instead of GIT
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print("Scene description model loaded!")
        
        print(f"Verified Model Precision: {next(self.model.parameters()).dtype}")

    def get_scene_description(self, frame):
        try:
            start_ai = time.time() ## Measure how long the AI takes to think
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

            end_ai = time.time()
            ai_time = end_ai - start_ai
            
            return caption,ai_time

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
    
    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n--- New Session Started: {time.ctime()} ---\n")
    
    # --- FPS VARIABLES ---
    prev_time = 0

    print("\nPress 'd' to get scene description")
    print("Press 'q' to quit")
    print("\nCamera is now active...")

    while True:
        curr_time = time.time()
        # Capture frame-by-frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break
        
        time_gap = curr_time - prev_time
        fps = 1 / time_gap if time_gap > 0 else 0
        prev_time = curr_time
        
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if int(time.time()) % 2 == 0: # Log roughly every 2 seconds
             log_file.write(f"Timestamp: {time.time()} | FPS: {int(fps)}\n")

        # Display the frame
        cv2.imshow('Scene Description', frame)

        # Wait for key press
        key = cv2.waitKey(1) & 0xFF

        # If 'd' is pressed, get scene description
        if key == ord('d'):
            print("AI is thinking...")
            description, duration = describer.get_scene_description(frame)
            print(f"Result: {description} (Time: {duration:.2f}s)")
            
            log_file.write(f"AI DESCRIPTION: {description} | Latency: {duration:.2f}s\n")


        # If 'q' is pressed, quit
        elif key == ord('q'):
            break

    # Release everything when done
    log_file.write("--- Session Ended ---\n")
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
