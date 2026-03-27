from transformers import AutoModelForCausalLM, AutoProcessor
import torch, cv2
from PIL import Image

class IsaacVLM:
    def __init__(self, model_id="PerceptronAI/Isaac-0.1"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.processor = AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

        self.model.eval()

    @torch.no_grad()
    def describe(self, frame_bgr, prompt: str, hint: str = "NONE") -> str:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }]

        inputs = self.processor(
            text=messages,
            return_tensors="pt",
        ).to(self.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

        return self.processor.batch_decode(
            output_ids, skip_special_tokens=True
        )[0].strip()
