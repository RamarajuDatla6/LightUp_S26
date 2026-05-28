import cv2
import torch
import time
import os
import psutil
from PIL import Image
# Swapped to BLIP-2 Classes
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class SceneDescriber:
    def __init__(self):
        # 1. UPDATED MODEL NAME (Evaluation Target)
        self.model_name = "Salesforce/blip2-opt-2.7b"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else "System CPU"
        
        print(f"Hardware Detected: {self.device_name} ({self.device})")
        print(f"Loading Model: {self.model_name}...")


        # 2. LOAD BLIP-2 
        # Note: BLIP-2 is large (~7GB-15GB VRAM/RAM). 
        # On GPU we use float16 to make it fit. On CPU we use float32.
        self.dtype = torch.float16 if self.device.type == 'cuda' else torch.float32
        
        self.processor = Blip2Processor.from_pretrained(self.model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            self.model_name, 
            torch_dtype=self.dtype
        ).to(self.device)
        
        self.actual_dtype = next(self.model.parameters()).dtype
        print(f"Model Load Complete. Precision: {self.actual_dtype}")

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024) 
        cpu_p = psutil.cpu_percent(interval=None) 
        return ram_mb, cpu_p

    def get_scene_description(self, frame):
        try:
            start_ai = time.time() 

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            
            # BLIP-2 specific input processing
            inputs = self.processor(image, return_tensors="pt").to(self.device, self.dtype)

            # Generate (keeping parameters simple for comparison)
            out = self.model.generate(**inputs, max_new_tokens=50)
            caption = self.processor.decode(out[0], skip_special_tokens=True).strip()
            
            if caption:
                caption = caption[0].upper() + caption[1:]
                if not caption.endswith('.'):
                    caption += '.'

            ai_time = time.time() - start_ai
            ram, cpu = self.get_system_usage()
            
            return caption, ai_time, ram, cpu

        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0

def main():
    try:
        describer = SceneDescriber()
    except Exception as e:
        print(f"Initialization Failed: {str(e)}")
        print("Tip: BLIP-2 requires significant RAM/VRAM. Ensure your system has 16GB+ or a strong GPU.")
        return

    cap = cv2.VideoCapture(0)
    # Keeping your 720p settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    res_label = f"{actual_w}x{actual_h}"

    if not cap.isOpened():
        print("Error: Camera Access Denied")
        return
    
    # Keeping your exact logging format
    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"SESSION START: {time.ctime()}\n")
    log_file.write(f"MODEL: {describer.model_name}\n")
    log_file.write(f"HARDWARE: {describer.device_name} ({describer.device})\n")
    log_file.write(f"RESOLUTION: {res_label} | PRECISION: {describer.actual_dtype}\n")
    log_file.write(f"{'-'*60}\n")
    
    prev_time = 0

    while True:
        curr_time = time.time()
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time
        
        # Keeping your on-screen UI
        cv2.putText(frame, f"FPS: {int(fps)} | {res_label}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.putText(frame, f"Device: {describer.device_name}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Member 3 - BLIP-2 EVALUATION', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            print("BLIP-2 processing...")
            desc, duration, ram, cpu = describer.get_scene_description(frame)
            
            # Log entry (Format identical to your BLIP-1 logs)
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