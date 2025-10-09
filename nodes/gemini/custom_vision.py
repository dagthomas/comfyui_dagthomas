# Gemini Custom Vision Node

import os
import re
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai

from ...utils.constants import CUSTOM_CATEGORY, gemini_models
from ...utils.image_utils import tensor2pil, pil2tensor


class GeminiCustomVision:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=self.gemini_api_key)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "gemini_model": (gemini_models,),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        fade_width = int(images[0].width * (fade_percentage / 100))
        total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
        max_height = max(img.height for img in images)

        combined_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for i, img in enumerate(images):
            if i == 0:
                combined_image.paste(img, (0, 0))
                x_offset = img.width - fade_width
            else:
                for x in range(fade_width):
                    factor = x / fade_width
                    for y in range(max_height):
                        if y < images[i - 1].height and y < img.height:
                            pixel1 = images[i - 1].getpixel(
                                (images[i - 1].width - fade_width + x, y)
                            )
                            pixel2 = img.getpixel((x, y))
                            blended_pixel = tuple(
                                int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                for c in range(3)
                            )
                            combined_image.putpixel((x_offset + x, y), blended_pixel)

                combined_image.paste(
                    img.crop((fade_width, 0, img.width, img.height)),
                    (x_offset + fade_width, 0),
                )
                x_offset += img.width - fade_width

        return combined_image

    def analyze_images(
        self,
        images,
        custom_prompt="",
        additive_prompt="",
        tag="",
        sex="other",
        pronouns="them, their",
        dynamic_prompt=False,
        words="100",
        fade_percentage=15.0,
        gemini_model="gemini-flash-latest",
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [tensor2pil(img) for img in images]
            else:
                pil_images = [tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)

            response = model.generate_content([full_prompt, combined_image])

            result = response.text

            faded_image_tensor = pil2tensor(combined_image)

            return (
                result,
                self.extract_first_two_sentences(result),
                faded_image_tensor,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], pil2tensor(error_image))
