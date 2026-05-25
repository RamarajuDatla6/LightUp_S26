import cv2
import torch
import time
import os
import psutil  # New: For RAM and CPU monitoring
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        # 1. TRACK THE MODEL NAME
        self.model_name = "Salesforce/blip-image-captioning-base"
        
        # 2. HARDWARE DETECTION
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else "System CPU"
        
        print(f"Hardware Detected: {self.device_name} ({self.device})")
        
        # Load Model
        print(f"Loading Model: {self.model_name}...")
        self.processor = BlipProcessor.from_pretrained(self.model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(self.model_name).to(self.device)
        
        # Technical Metric: Precision
        self.dtype = next(self.model.parameters()).dtype
        print(f"Model Load Complete. Precision: {self.dtype}")

    def get_system_usage(self):
        """Helper function to capture hardware metrics"""
        process = psutil.Process(os.getpid())
        # RAM usage in Megabytes
        ram_mb = process.memory_info().rss / (1024 * 1024) 
        # CPU usage percentage (of this specific process)
        cpu_p = psutil.cpu_percent(interval=None) 
        return ram_mb, cpu_p

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
            
            # Capture hardware stats immediately after inference
            ram, cpu = self.get_system_usage()
            
            return caption, ai_time, ram, cpu

        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0

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
    
    # --- DETAILED LOGGING ---
    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"SESSION START: {time.ctime()}\n")
    log_file.write(f"MODEL: {describer.model_name}\n")
    log_file.write(f"HARDWARE: {describer.device_name} ({describer.device})\n")
    log_file.write(f"RESOLUTION: {res_label} | PRECISION: {describer.dtype}\n")
    log_file.write(f"{'-'*60}\n")
    
    prev_time = 0

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # --- ON-SCREEN METRICS ---
        cv2.putText(frame, f"FPS: {int(fps)} | {res_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Device: {describer.device_name}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Member 3 - HD System Auditor', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            print("AI Processing...")
            desc, duration, ram, cpu = describer.get_scene_description(frame)
            
            # Log Result with System Metrics
            log_entry = (f"[{time.strftime('%H:%M:%S')}] {desc}\n"
                         f" > Latency: {duration:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}% | Device: {describer.device}\n")
            
            log_file.write(log_entry)
            print(log_entry)

        elif key == ord('q'):
            break

    log_file.write(f"SESSION END: {time.ctime()}\n")
    log_file.close()
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()