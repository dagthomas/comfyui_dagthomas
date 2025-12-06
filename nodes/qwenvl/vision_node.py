# QwenVL Vision Node

import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
import torch
import gc
from pathlib import Path
# huggingface_hub not needed - transformers handles downloads automatically
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models
import folder_paths


class QwenVLVisionNode:
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
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct"}),
                "max_tokens": ("INT", {"default": 512, "min": 64, "max": 2048}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_images"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    @staticmethod
    def tensor_to_pil(img_tensor):
        """Convert tensor to PIL image"""
        if img_tensor.dim() == 4:
            img_tensor = img_tensor[0]
        i = 255.0 * img_tensor.cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    @classmethod
    def ensure_model(cls, model_name):
        """Get model path and cache directory"""
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Handle full repo paths (e.g., "huihui-ai/Model-Name") vs simple names (e.g., "Qwen3-VL-4B")
        if "/" in model_name:
            repo_id = model_name  # Use as-is
            folder_name = model_name.replace("/", "--")  # Safe folder name
        else:
            repo_id = f"Qwen/{model_name}"  # Default Qwen namespace
            folder_name = model_name

        target = models_dir / folder_name

        # Check if model files exist locally (manually placed)
        if target.exists():
            safetensors = list(target.glob("*.safetensors"))
            bins = list(target.glob("*.bin"))
            config = target / "config.json"
            if (safetensors or bins) and config.exists():
                print(f"📁 Using local model: {target}")
                return str(target), None  # No cache_dir needed for local

        # Return repo_id and cache_dir for downloading to local folder
        print(f"📥 Will download {repo_id} to {models_dir}")
        return repo_id, str(models_dir)

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
        model_path, cache_dir = cls.ensure_model(model_name)

        print(f"🔄 Loading {model_name}...")

        # Determine device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        # Build kwargs - add cache_dir if downloading
        load_kwargs = {
            "device_map": {"": 0} if device == "cuda" else device,
            "dtype": dtype,
            "attn_implementation": "sdpa",
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        # Load model
        model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

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

    def analyze_images(
        self,
        images,
        happy_talk,
        compress,
        compression_level,
        poster,
        qwen_model="Qwen3-VL-4B-Instruct",
        max_tokens=512,
        temperature=0.7,
        keep_model_loaded=True,
        custom_base_prompt="",
        custom_title="",
        override="",
    ):
        try:
            # Build prompt
            if override:
                prompt_text = override
            else:
                prompt_text = self._build_prompt(
                    happy_talk, compress, compression_level, poster,
                    custom_base_prompt, custom_title
                )

            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded)

            # Prepare conversation
            conversation = [{"role": "user", "content": []}]

            # Add images
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                conversation[0]["content"].append({"type": "image", "image": pil_image})

            # Add text prompt
            conversation[0]["content"].append({"type": "text", "text": prompt_text})

            # Apply chat template
            chat = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Get images for processing
            images_list = [item["image"] for item in conversation[0]["content"] if item["type"] == "image"]

            # Process inputs
            processed = processor(text=chat, images=images_list or None, return_tensors="pt")

            # Move to device
            model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }

            # Generate
            print(f"🔄 QwenVL ({qwen_model}): Generating response...")

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

            # Decode
            input_len = model_inputs["input_ids"].shape[-1]
            result = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            print(f"✅ QwenVL ({qwen_model}): Generated {len(result)} characters")
            if len(result) < 200:
                print(f"📝 Full response: {result}")
            else:
                print(f"📝 First 200 chars: {result[:200]}")

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            return (result,)

        except Exception as e:
            import traceback
            error_msg = f"QwenVL Error: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())

            # Clear model on error
            if not keep_model_loaded:
                self.clear_model()

            return (error_msg,)

    def _build_prompt(self, happy_talk, compress, compression_level, poster, custom_base_prompt, custom_title):
        """Build prompt based on settings"""
        if custom_base_prompt.strip():
            return custom_base_prompt.strip()

        default_happy_prompt = """Analyze the provided images and create a detailed visually descriptive caption that combines elements from all images into a single cohesive composition. Imagine all images being movie stills from real movies. This caption will be used as a prompt for a text-to-image AI system. Focus on:
1. Detailed visual descriptions of characters, including ethnicity, skin tone, expressions, etc.
2. Overall scene and background details.
3. Image style, photographic techniques, direction of photo taken.
4. Cinematography aspects with technical details.
5. If multiple characters are present, describe an interesting interaction between two primary characters.
6. Incorporate a specific movie director's visual style (e.g., Wes Anderson, Christopher Nolan, Quentin Tarantino).
7. Describe the lighting setup in detail, including type, color, and placement of light sources.

If you want to add text, state the font, style and where it should be placed, a sign, a poster, etc. The text should be captioned like this " ".
Always describe how characters look at each other, their expressions, and the overall mood of the scene.

ALWAYS remember to note that it is a cinematic movie still and describe the film grain, color grading, and any artifacts or characteristics specific to film photography.
ALWAYS create the output as one scene, never transition between scenes."""

        default_simple_prompt = """Analyze the provided images and create a brief, straightforward caption that combines key elements from all images. Focus on the main subjects, overall scene, and atmosphere. Provide a clear and concise description in one or two sentences, suitable for a text-to-image AI system."""

        poster_prompt = f"""Analyze the provided images and extract key information to create a cinematic movie poster style description. Format the output as follows:

Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters."""

        if poster:
            base_prompt = poster_prompt
        else:
            base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

        if compress and not poster:
            compression_chars = {
                "soft": 600 if happy_talk else 300,
                "medium": 400 if happy_talk else 200,
                "hard": 200 if happy_talk else 100,
            }
            char_limit = compression_chars[compression_level]
            base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

        return base_prompt
