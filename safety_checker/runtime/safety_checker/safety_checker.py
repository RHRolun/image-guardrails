import kserve
import numpy as np
import logging
import base64
from io import BytesIO

from diffusers import StableDiffusionXLPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor, CLIPTokenizer, CLIPModel
import torch
from torchvision.transforms.functional import pil_to_tensor, to_pil_image
from PIL import Image as PIL_Image

logging.basicConfig(level=logging.INFO)

class SafetyChecker(kserve.Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.ready = True

        #self.device="cuda"
        self.device="cpu"
        dtype=torch.float16

        self.feature_extractor = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker").to(self.device)

    def predict(self, inputs: dict) -> dict:
        try:
            image_data = inputs.get("image")
            if not image_data:
                return {"error": "Missing image data"}

            # Decode the base64 image
            image_bytes = base64.b64decode(image_data)
            image = PIL_Image.open(BytesIO(image_bytes)).convert("RGB")

            safety_checker_input = self.feature_extractor(image, return_tensors="pt").to(self.device)
            image, has_nsfw_concept = self.safety_checker(
                images=pil_to_tensor(image).unsqueeze(0), clip_input=safety_checker_input.pixel_values.to(dtype)
            )
            return {"predictions": has_nsfw_concept}

        except Exception as e:
            logging.error(f"Prediction failed: {e}")
            return {"error": str(e)}