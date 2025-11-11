# Grok Vision Node

import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI
import httpx

from ...utils.constants import CUSTOM_CATEGORY, grok_models


class GrokVisionNode:
    def __init__(self):
        api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
        if not api_key:
            raise ValueError("XAI_API_KEY or GROK_API_KEY environment variable is not set")

        # Create a compatible httpx client to avoid version conflicts
        try:
            http_client = httpx.Client(timeout=60.0)
        except TypeError:
            # Fallback for older httpx versions that don't support certain parameters
            http_client = httpx.Client()

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.x.ai/v1",
            http_client=http_client
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
                "grok_model": (["grok-2-vision-1212", "grok-4-0709", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning"], {"default": "grok-2-vision-1212"}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_images"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def tensor_to_pil(self, img_tensor):
        i = 255.0 * img_tensor.cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def analyze_images(
        self,
        images,
        happy_talk,
        compress,
        compression_level,
        poster,
        grok_model="grok-2-vision-1212",
        custom_base_prompt="",
        custom_title="",
        override="",
    ):
        # Process first image
        image = self.tensor_to_pil(images[0])
        base64_image = self.encode_image(image)
        
        # Build the prompt
        if override:
            prompt = override
        else:
            if custom_base_prompt:
                base_prompt = custom_base_prompt
            else:
                base_prompt = self._get_base_prompt(happy_talk, compress, compression_level, poster)
            
            if custom_title:
                prompt = f"{base_prompt}\n\nTitle: {custom_title}\n\nAnalyze this image and provide a detailed description."
            else:
                prompt = f"{base_prompt}\n\nAnalyze this image and provide a detailed description."

        try:
            response = self.client.chat.completions.create(
                model=grok_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.7,
            )
            
            result = response.choices[0].message.content.strip()
            print(f"🤖 Grok Vision ({grok_model}) analyzed image: {len(result)} characters")
            
            return (result,)
            
        except Exception as e:
            error_msg = f"Grok Vision API Error: {str(e)}"
            print(f"❌ {error_msg}")
            return (error_msg,)

    def _get_base_prompt(self, happy_talk, compress, compression_level, poster):
        """Generate base prompt based on settings"""
        base_prompt = "You are Grok, an AI assistant with vision capabilities. "
        
        if happy_talk:
            base_prompt += "Describe the image in an enthusiastic, positive, and engaging tone. "
        else:
            base_prompt += "Describe the image in a clear, direct, and professional tone. "
        
        if compress:
            if compression_level == "soft":
                base_prompt += "Keep the description concise while maintaining key visual details. "
            elif compression_level == "medium":
                base_prompt += "Provide a moderately compressed description focusing on essential visual elements. "
            elif compression_level == "hard":
                base_prompt += "Give a very brief description with only the most important visual aspects. "
        
        if poster:
            base_prompt += "Format the description as if it were promotional or poster-style content with impactful language. "
        
        return base_prompt
