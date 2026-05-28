import cv2
import torch
import time
import os
import psutil
from PIL import Image

# ============================================================
# TECHNICAL PATCHES (Required for Moondream + Python 3.13)
# ============================================================
import transformers
if not hasattr(transformers.PreTrainedModel, "all_tied_weights_keys"):
    @property
    def fake_tied_keys(self): return {}
    transformers.PreTrainedModel.all_tied_weights_keys = fake_tied_keys

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# ============================================================

class SceneDescriber:
    def __init__(self):
        # 1. TRACK THE MODEL NAME
        self.model_name = "vikhyatk/moondream2"
        
        # 2. HARDWARE DETECTION
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device_name = torch.cuda.get_device_name(0) if self.device.type == 'cuda' else "System CPU"
        
        print(f"Hardware Detected: {self.device_name} ({self.device})")
        print(f"Loading Model: {self.model_name}...")

        # 3. LOAD MOONDREAM WITH COMPATIBILITY PATCHES
        try:
            config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
            # Fix for missing attributes in newer library versions
            if not hasattr(config, "pad_token_id"): config.pad_token_id = 0
            if not hasattr(config, "eos_token_id"): config.eos_token_id = 0
            
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, 
                trust_remote_code=True, 
                config=config,
                low_cpu_mem_usage=True
            ).to(self.device)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Technical Metric: Precision
            self.dtype = next(self.model.parameters()).dtype
            print(f"Model Load Complete. Precision: {self.dtype}")
        except Exception as e:
            print(f"Initialization Failed: {e}")
            raise e

    def get_system_usage(self):
        """Helper function to capture hardware metrics"""
        process = psutil.Process(os.getpid())
        ram_mb = process.memory_info().rss / (1024 * 1024) 
        cpu_p = psutil.cpu_percent(interval=None) 
        return ram_mb, cpu_p

    def get_scene_description(self, frame):
        try:
            start_ai = time.time() 

            # Moondream Processing
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_rgb)
            
            # Step 1: Scan/Encode the image
            image_embeds = self.model.encode_image(image)
            
            # Step 2: Answer the question (Generation)
            caption = self.model.answer_question(image_embeds, "Describe this image.", self.tokenizer)

            ai_time = time.time() - start_ai
            
            # Capture hardware stats immediately after inference
            ram, cpu = self.get_system_usage()
            
            return caption.strip(), ai_time, ram, cpu

        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0

def main():
    try:
        describer = SceneDescriber()
    except Exception as e:
        print(f"Failed to start Moondream: {e}")
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
    
    # --- DETAILED LOGGING (Same format as BLIP-1 and BLIP-2) ---
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
        
        cv2.putText(frame, f"Model: MOONDREAM2", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Member 3 - Moondream Evaluation', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('d'):
            print("Moondream is thinking...")
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