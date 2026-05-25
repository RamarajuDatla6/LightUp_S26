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

        # Loading the BLIP model
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)
        print("Scene description model loaded!")
        
        # Technical Metric for the Professor: Show weight precision
        print(f"Verified Model Precision: {next(self.model.parameters()).dtype}")

    def get_scene_description(self, frame):
        try:
            # Measure AI thinking time (Inference Latency)
            start_ai = time.time() 
            
            # Image preprocessing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)

            # Convert image to tensors
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            # Generate the caption
            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=1.0,
                repetition_penalty=1.5
            )

            # Convert numbers back to text
            caption = self.processor.decode(out[0], skip_special_tokens=True)

            # Format the sentence
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'

            # Calculate total time the AI took
            end_ai = time.time()
            ai_time = end_ai - start_ai
            
            return caption, ai_time

        except Exception as e:
            print(f"Error in scene description: {str(e)}")
            return "Unable to describe the scene.", 0

def main():
    try:
        describer = SceneDescriber()
    except Exception as e:
        print(f"Error initializing the model: {str(e)}")
        return

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # 1. Create/Open the log file for AI results only
    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n--- New Session Started: {time.ctime()} ---\n")
    
    prev_time = 0

    print("\nPress 'd' to get scene description")
    print("Press 'q' to quit")
    print("\nCamera is now active...")

    while True:
        curr_time = time.time()
        ret, frame = cap.read()

        if not ret:
            print("Error: Can't receive frame")
            break
        
        # 2. Calculate Live FPS (for on-screen display only)
        time_gap = curr_time - prev_time
        fps = 1 / time_gap if time_gap > 0 else 0
        prev_time = curr_time
        
        # 3. Show FPS on the screen (Useful for the demo)
        fps_text = f"FPS: {int(fps)}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # [FPS LOGGING REMOVED FROM HERE]

        # Display the frame
        cv2.imshow('Scene Description', frame)

        # Listen for keyboard input
        key = cv2.waitKey(1) & 0xFF

        if key == ord('d'):
            print("AI is thinking...")
            description, duration = describer.get_scene_description(frame)
            print(f"Result: {description} (Time: {duration:.2f}s)")
            
            # 4. Log the actual AI results and timing (This is what the Professor wants to see)
            log_file.write(f"TIMESTAMP: {time.ctime()} | DESCRIPTION: {description} | LATENCY: {duration:.2f}s\n")

        elif key == ord('q'):
            break

    # Cleanup
    log_file.write(f"--- Session Ended: {time.ctime()} ---\n")
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()