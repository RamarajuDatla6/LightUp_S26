import cv2
import torch
import time
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        # 1. TRACK THE MODEL NAME
        self.model_name = "Salesforce/blip-image-captioning-base"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Executing on: {self.device}")
        
        # Load Model
        print(f"Loading Model: {self.model_name}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        
        # Technical Metric: Precision
        self.dtype = next(self.model.parameters()).dtype
        print(f"Model Load Complete. Precision: {self.dtype}")

    def get_scene_description(self, frame):
        try:
            start_ai = time.time() 

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            out = self.model.generate(
                **inputs,
                max_length=50,
                num_beams=5,
                temperature=1.0,
                repetition_penalty=1.5
            )

            caption = self.processor.decode(out[0], skip_special_tokens=True)
            caption = caption[0].upper() + caption[1:]
            if not caption.endswith('.'):
                caption += '.'

            ai_time = time.time() - start_ai
            return caption, ai_time

        except Exception as e:
            return f"Error: {str(e)}", 0

def main():
    try:
        describer = SceneDescriber()
    except Exception as e:
        print(f"Initialization Failed: {str(e)}")
        return

    # --- CAMERA INITIALIZATION ---
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res_label = f"{actual_w}x{actual_h}"

    if not cap.isOpened():
        print("Error: Access Denied to Webcam")
        return
    
    # --- UPDATED LOGGING ---
    # We now log the MODEL NAME at the start of the session
    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n--- SESSION: {time.ctime()} ---\n")
    log_file.write(f"MODEL: {describer.model_name}\n")
    log_file.write(f"RESOLUTION: {res_label} | PRECISION: {describer.dtype}\n")
    log_file.write("-" * 50 + "\n")
    
    prev_time = 0
    print(f"\nModel {describer.model_name} is active at {res_label}.")

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        # Calculate Live FPS
        time_gap = curr_time - prev_time
        fps = 1 / time_gap if time_gap > 0 else 0
        prev_time = curr_time
        
        # --- ON-SCREEN METRICS ---
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Mode: {res_label} HD", (20, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Adding Model Name to the screen as well (Optional but looks professional)
        cv2.putText(frame, f"AI: {describer.model_name.split('/')[-1]}", (20, 125), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Member 3 - High Definition Scene Description', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            print(f"AI ({describer.model_name}) is analyzing...")
            description, duration = describer.get_scene_description(frame)
            
            # Print to console and save to Log
            log_entry = f"[{time.strftime('%H:%M:%S')}] {description} | Latency: {duration:.2f}s\n"
            log_file.write(log_entry)
            print(log_entry)

        elif key == ord('q'):
            break

    # Shutdown
    log_file.write(f"--- Session End: {time.ctime()} ---\n")
    log_file.write("=" * 50 + "\n")
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()