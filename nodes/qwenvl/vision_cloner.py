# QwenVL Vision Cloner Node

import os
import json
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models
import folder_paths


class QwenVLVisionCloner:
    # Class-level cache for model
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_model_name = None

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct"}),
                "max_tokens": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""})
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "formatted_output",
        "raw_json",
        "faded_image",
    )
    FUNCTION = "analyze_images"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    @staticmethod
    def tensor2pil(image):
        """Convert tensor to PIL image"""
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        """Convert PIL image to tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        """Fade multiple images together"""
        if len(images) < 2:
            return images[0] if images else None

        # Determine orientation based on aspect ratio
        aspect_ratio = images[0].width / images[0].height
        vertical_stack = aspect_ratio > 1

        if vertical_stack:
            # Vertical stacking for wider images
            fade_height = int(images[0].height * (fade_percentage / 100))
            total_height = sum(img.height for img in images) - fade_height * (
                len(images) - 1
            )
            max_width = max(img.width for img in images)
            combined_image = Image.new("RGB", (max_width, total_height))

            y_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    y_offset = img.height - fade_height
                else:
                    for y in range(fade_height):
                        factor = y / fade_height
                        for x in range(max_width):
                            if x < images[i - 1].width and x < img.width:
                                pixel1 = images[i - 1].getpixel(
                                    (x, images[i - 1].height - fade_height + y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x, y_offset + y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((0, fade_height, img.width, img.height)),
                        (0, y_offset + fade_height),
                    )
                    y_offset += img.height - fade_height
        else:
            # Horizontal stacking for taller images
            fade_width = int(images[0].width * (fade_percentage / 100))
            total_width = sum(img.width for img in images) - fade_width * (
                len(images) - 1
            )
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
                                combined_image.putpixel(
                                    (x_offset + x, y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((fade_width, 0, img.width, img.height)),
                        (x_offset + fade_width, 0),
                    )
                    x_offset += img.width - fade_width

        return combined_image

    def format_element(self, element):
        """Format a single element from the JSON response"""
        element_type = element.get("Type", "")
        description = element.get("Description", "").lower()
        attributes = element.get("Attributes", {})

        if element_type.lower() == "object":
            formatted = description
        else:
            formatted = f"{element_type} {description}"

        attr_details = []
        for key, value in attributes.items():
            if value and value.lower() not in ["n/a", "none", "None"]:
                attr_details.append(f"{key.lower()}: {value.lower()}")

        if attr_details:
            formatted += f" ({', '.join(attr_details)})"

        return formatted

    def extract_data(self, data):
        """Extract and format data from JSON response"""
        extracted = []
        extracted.append(data.get("Title", ""))
        extracted.append(data.get("Artistic Style", ""))

        color_scheme = data.get("Color Scheme", [])
        if color_scheme:
            extracted.append(
                f"palette ({', '.join(color.lower() for color in color_scheme)})"
            )

        for element in data.get("Elements", []):
            extracted.append(self.format_element(element))

        overall_scene = data.get("Overall Scene", {})
        extracted.append(overall_scene.get("Theme", ""))
        extracted.append(overall_scene.get("Setting", ""))
        extracted.append(overall_scene.get("Lighting", ""))

        return ", ".join(item for item in extracted if item)

    @classmethod
    def ensure_model(cls, model_name):
        """Download model if not present"""
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Map display name to repo ID
        repo_id = f"Qwen/{model_name}"
        target = models_dir / model_name

        if not target.exists():
            print(f"📥 Downloading {model_name}...")
            snapshot_download(
                repo_id=repo_id,
                local_dir=str(target),
                ignore_patterns=["*.md", ".git*"],
            )
            print(f"✅ Downloaded {model_name}")

        return str(target)

    @classmethod
    def load_model(cls, model_name, keep_loaded=True):
        """Load model with caching support"""
        # Check if model is already loaded
        if keep_loaded and cls._cached_model is not None and cls._cached_model_name == model_name:
            print(f"♻️ Using cached {model_name}")
            return cls._cached_model, cls._cached_processor, cls._cached_tokenizer

        # Clear old model if loading a different one
        if cls._cached_model_name != model_name:
            cls.clear_model()

        # Download/locate model
        model_path = cls.ensure_model(model_name)

        print(f"🔄 Loading {model_name}...")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Load model
        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map={"": 0} if device == "cuda" else device,
            dtype=dtype,
            attn_implementation="sdpa",
            use_safetensors=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # Cache if requested
        if keep_loaded:
            cls._cached_model = model
            cls._cached_processor = processor
            cls._cached_tokenizer = tokenizer
            cls._cached_model_name = model_name

        print(f"✅ Loaded {model_name}")
        return model, processor, tokenizer

    @classmethod
    def clear_model(cls):
        """Clear cached model"""
        if cls._cached_model is not None:
            print(f"🧹 Clearing cached model")
            cls._cached_model = None
            cls._cached_processor = None
            cls._cached_tokenizer = None
            cls._cached_model_name = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def analyze_images(self, images, fade_percentage=15.0, qwen_model="Qwen3-VL-4B-Instruct",
                      max_tokens=4096, temperature=0.7, keep_model_loaded=True, custom_prompt=""):
        try:
            default_prompt = """You must respond with ONLY valid JSON, nothing else. Analyze the image and output a JSON object:

{
  "title": "descriptive title",
  "artistic_style": "art style description",
  "color_scheme": ["color1", "color2", "color3"],
  "elements": [
    {
      "type": "character or object",
      "description": "brief description",
      "attributes": {
        "clothing": "if applicable",
        "position": "location in scene",
        "other": "relevant details"
      }
    }
  ],
  "overall_scene": {
    "theme": "main theme",
    "setting": "location and environment",
    "lighting": "lighting description",
    "mood": "emotional tone"
  }
}

CRITICAL: Output ONLY the JSON object. No explanations, no markdown code blocks, no extra text. Just pure JSON starting with { and ending with }."""

            final_prompt = custom_prompt if custom_prompt.strip() else default_prompt

            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded)

            # Handle single image or multiple images
            if len(images.shape) == 3:  # Single image
                pil_images = [self.tensor2pil(images)]
            else:  # Multiple images
                pil_images = [self.tensor2pil(img) for img in images]

            # Fade images together
            combined_image = self.fade_images(pil_images, fade_percentage)

            # Prepare conversation
            conversation = [{"role": "user", "content": []}]
            conversation[0]["content"].append({"type": "image", "image": combined_image})
            conversation[0]["content"].append({"type": "text", "text": final_prompt})

            # Apply chat template
            chat = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            processed = processor(text=chat, images=[combined_image], return_tensors="pt")

            # Move to device
            model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }

            print(f"🔄 QwenVL Vision Cloner: Sending request to {qwen_model}...")

            # Generate
            stop_tokens = [tokenizer.eos_token_id]
            if hasattr(tokenizer, "eot_id") and tokenizer.eot_id is not None:
                stop_tokens.append(tokenizer.eot_id)

            with torch.no_grad():
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=0.9,
                    eos_token_id=stop_tokens,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            print(f"✅ QwenVL Vision Cloner: Received response from {qwen_model}")

            # Decode
            input_len = model_inputs["input_ids"].shape[-1]
            content = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            # Debug: Print the raw response
            print(f"📝 Raw response length: {len(content)} characters")
            print(f"📝 First 200 chars: {content[:200]}")

            if not content:
                print("⚠️ Model returned empty response")
                error_image = Image.new("RGB", (512, 512), color="red")
                if not keep_model_loaded:
                    self.clear_model()
                return ("Model returned empty response", "{}", self.pil2tensor(error_image))

            # Try to extract JSON from the response
            json_str = content

            # Check if the content is wrapped in Markdown code blocks
            if "```json" in content:
                start = content.find("```json") + 7
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
            elif "```" in content:
                start = content.find("```") + 3
                end = content.find("```", start)
                if end != -1:
                    json_str = content[start:end].strip()
            else:
                # Try to find JSON object boundaries
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1].strip()

            # Parse the JSON
            data = json.loads(json_str)

            # Handle single image or multiple images
            if isinstance(data, list):
                results = [self.extract_data(image_data) for image_data in data]
                result = " | ".join(results)
            else:
                result = self.extract_data(data)

            faded_image_tensor = self.pil2tensor(combined_image)

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            return (result, json.dumps(data, indent=2), faded_image_tensor)

        except json.JSONDecodeError as e:
            print(f"❌ QwenVL Vision Cloner: Failed to parse JSON response: {e}")
            print(f"📝 Content that failed to parse: {content if 'content' in locals() else 'N/A'}")

            # If we have content but it's not JSON, return it as the formatted output
            if 'content' in locals() and content:
                print("⚠️ Returning raw text response since JSON parsing failed")
                faded_image_tensor = self.pil2tensor(combined_image)
                error_json = json.dumps({"error": "JSON parse failed", "raw_response": content[:1000]}, indent=2)
                if not keep_model_loaded:
                    self.clear_model()
                return (content, error_json, faded_image_tensor)

            error_message = f"Invalid JSON response from model: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            if not keep_model_loaded:
                self.clear_model()
            return (error_message, "{}", self.pil2tensor(error_image))

        except Exception as e:
            import traceback
            print(f"❌ QwenVL Vision Cloner: Unexpected error: {e}")
            print(traceback.format_exc())
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            if not keep_model_loaded:
                self.clear_model()
            return (error_message, "{}", self.pil2tensor(error_image))
