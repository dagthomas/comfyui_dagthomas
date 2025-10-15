# OllamaVisionNode Node

from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil, pil2tensor
from PIL import Image
import base64
import io
import json
import numpy as np
import re
import requests
import torch


class OllamaVisionNode:
    def __init__(self):
        self.loaded_model = None

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
                "custom_model": ("STRING", {"default": "llava-llama3:latest"}),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434/api/generate"},
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "analyze_images"
    CATEGORY = "dagthomas"

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

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

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def check_ollama_service(self, url):
            try:
                response = requests.get(url.replace("/api/generate", ""))
                return response.status_code == 200
            except requests.RequestException:
                return False
            
    def check_and_load_model(self, model_name, ollama_url):
        if self.loaded_model != model_name:
            load_url = ollama_url.replace("/api/generate", "/api/pull")
            try:
                response = requests.post(load_url, json={"name": model_name})
                if response.status_code == 200:
                    print(f"Successfully loaded model: {model_name}")
                    self.loaded_model = model_name
                else:
                    raise Exception(f"Failed to load model: {model_name}. Status code: {response.status_code}")
            except requests.RequestException as e:
                raise Exception(f"Error while loading model: {e}")

    def unload_model(self, ollama_url):
        """
        Mark model as no longer actively loaded.
        Note: Ollama will automatically manage VRAM and unload models when needed.
        We do NOT call /api/delete as that would delete the model entirely from disk.
        """
        if self.loaded_model:
            print(f"🔓 Marking model as inactive: {self.loaded_model}")
            print("💡 Ollama will automatically manage VRAM (model stays on disk)")
            self.loaded_model = None

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
        custom_model="llava-llama3:latest",
        ollama_url="http://localhost:11434/api/generate",
        unload_after_use=True,
    ):
        try:
            if not self.check_ollama_service(ollama_url):
                raise Exception("Ollama service is not running. Please start it with 'ollama serve'.")
            
            self.check_and_load_model(custom_model, ollama_url)

            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            encoded_image = self.encode_image(combined_image)

            payload = {
                "model": custom_model,
                "prompt": full_prompt,
                "images": [encoded_image],
                "stream": False
            }

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            faded_image_tensor = self.pil2tensor(combined_image)
            if unload_after_use:
                self.unload_model(ollama_url)

            return (
                result,
                self.extract_first_two_sentences(result),
                faded_image_tensor,
            )

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))

