import cv2
import torch
import time
import os
import psutil
from PIL import Image

# ============================================================
# TECHNICAL PATCHES
# ============================================================
import transformers.utils.import_utils as import_utils
if not hasattr(import_utils, "is_torch_fx_available"):
    import_utils.is_torch_fx_available = lambda: False

from transformers import AutoModelForCausalLM, AutoTokenizer
# ============================================================

class SceneDescriber:
    def __init__(self):
        self.model_name = "openbmb/MiniCPM-V-2"
        self.device = torch.device("cpu") # Force CPU to avoid 'type' errors on GPU mapping
        
        print(f"Loading Model: {self.model_name} in SAFE MODE...")

        try:
            # SAFE LOADING: We remove device_map and float16 which cause the 'type' error
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id if hasattr(self, 'model_id') else self.model_name, 
                trust_remote_code=True,
                low_cpu_mem_usage=True
            ).to(self.device).eval()
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            print(f"Model Load Complete.")
        except Exception as e:
            print(f"Initialization Failed: {e}")
            raise e

    def get_system_usage(self):
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024), psutil.cpu_percent(interval=None)

    def get_scene_description(self, frame):
        try:
            start_ai = time.time() 
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            # MiniCPM chat logic
            msgs = [{'role': 'user', 'content': 'Describe this image.'}]
            caption, _ = self.model.chat(image=image, msgs=msgs, tokenizer=self.tokenizer)

            ai_time = time.time() - start_ai
            ram, cpu = self.get_system_usage()
            return caption.strip(), ai_time, ram, cpu
        except Exception as e:
            return f"Error: {str(e)}", 0, 0, 0

def main():
    try:
        describer = SceneDescriber()
    except:
        return

    cap = cv2.VideoCapture(0)
    cap.set(3, 1280); cap.set(4, 720)

    log_file = open("performance_log.txt", "a")
    log_file.write(f"\n--- SESSION: {time.ctime()} | MODEL: MINICPM ---\n")
    
    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        fps = 1 / (time.time() - prev_time) if (time.time() - prev_time) > 0 else 0
        prev_time = time.time()
        
        cv2.putText(frame, f"FPS: {int(fps)} | MINICPM SAFE-MODE", (20, 40), 1, 2, (0, 255, 0), 2)
        cv2.imshow('MiniCPM Evaluation', frame)

        if cv2.waitKey(1) & 0xFF == ord('d'):
            print("Analyzing...")
            desc, duration, ram, cpu = describer.get_scene_description(frame)
            log_entry = f"[{time.strftime('%H:%M:%S')}] {desc}\n > Latency: {duration:.2f}s | RAM: {ram:.1f}MB | CPU: {cpu}%\n"
            print(log_entry)
            log_file.write(log_entry)
        elif cv2.waitKey(1) & 0xFF == ord('q'):
            break

    log_file.close()
    cap.release(); cv2.destroyAllWindows()

if __name__ == "__main__":
    main()