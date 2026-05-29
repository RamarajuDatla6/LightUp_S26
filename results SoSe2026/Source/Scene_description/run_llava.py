import cv2, torch, time, psutil, os
from PIL import Image
# We use the specific Llava classes instead of AutoModel
from transformers import LlavaForConditionalGeneration, LlavaProcessor

# SETTINGS
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 7B models are massive. float16 is MANDATORY if you have a GPU.
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print("Loading LLaVA-1.5-7B (The 15GB Space Shuttle)...")
model_id = "llava-hf/llava-1.5-7b-hf"

# Load the Correct Llava Classes
processor = LlavaProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=DTYPE, 
    low_cpu_mem_usage=True, 
    device_map="auto" if torch.cuda.is_available() else None
)

if not torch.cuda.is_available():
    model = model.to(DEVICE)

cap = cv2.VideoCapture(0); cap.set(3, 1280); cap.set(4, 720)

print("\n--- LLaVA READY ---")
print("Warning: This model is 7 Billion parameters. It will be VERY slow on a CPU.")

while True:
    ret, frame = cap.read()
    if not ret: break
    cv2.putText(frame, "MODEL: LLaVA-7B (720p)", (20, 40), 1, 1.5, (0, 255, 0), 2)
    cv2.imshow('Vision Audit', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('d'):
        print("LLaVA is thinking (Expect 30-60 seconds on CPU)...")
        start = time.time()
        
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        # LLaVA needs a specific prompt format to work correctly
        prompt = "USER: <image>\nDescribe this image in one short sentence. ASSISTANT:"
        
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(DEVICE, DTYPE)
        
        # Generate
        out = model.generate(**inputs, max_new_tokens=50, do_sample=False)
        txt = processor.decode(out[0], skip_special_tokens=True)
        
        # Clean the output (LLaVA repeats the prompt in the answer, we want just the Assistant part)
        answer = txt.split("ASSISTANT:")[-1].strip()
        
        lat = time.time() - start
        ram = psutil.Process(os.getpid()).memory_info().rss / (1024**2)
        
        log_entry = f"[{time.ctime()}] LLAVA: {answer} | {lat:.2f}s | {ram:.1f}MB\n"
        print(log_entry)
        open("performance_log.txt", "a").write(log_entry)
        
    elif cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release(); cv2.destroyAllWindows()