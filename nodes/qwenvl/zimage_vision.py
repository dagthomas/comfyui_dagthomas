# QwenVL Z-Image Vision Node
# Analyzes images and outputs in Z-Image TurnBuilder chat format

import json
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models, prompt_dir
import folder_paths
import os


class QwenVLZImageVision:
    """
    Analyzes images and outputs descriptions in Z-Image TurnBuilder format.
    Wraps image analysis in chat template tokens (<|im_start|>/<|im_end|>) for compatible systems.
    """
    
    # Class-level cache for model
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_model_name = None
    
    # Default prompt file names
    DEFAULT_PROMPT_FILE = "zimage_vision_analysis.txt"
    DEFAULT_SYSTEM_PROMPT_FILE = "zimage_vision_system.txt"
    DEFAULT_USER_MOD_FILE = "zimage_vision_user_mod.txt"

    def __init__(self):
        pass

    @staticmethod
    def get_prompt_files():
        """Get list of available prompt files from custom_prompts directory"""
        try:
            files = [f for f in os.listdir(prompt_dir) if f.endswith(".txt")]
            return ["(none)"] + sorted(files)
        except Exception:
            return ["(none)"]

    @classmethod
    def INPUT_TYPES(cls):
        prompt_files = cls.get_prompt_files()
        # Set defaults if files exist, otherwise (none)
        default_prompt = cls.DEFAULT_PROMPT_FILE if cls.DEFAULT_PROMPT_FILE in prompt_files else "(none)"
        default_system = cls.DEFAULT_SYSTEM_PROMPT_FILE if cls.DEFAULT_SYSTEM_PROMPT_FILE in prompt_files else "(none)"
        default_user_mod = cls.DEFAULT_USER_MOD_FILE if cls.DEFAULT_USER_MOD_FILE in prompt_files else "(none)"
        
        return {
            "required": {
                "images": ("IMAGE",),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct"}),
                "prompt_file": (prompt_files, {"default": default_prompt}),
                "system_prompt_file": (prompt_files, {"default": default_system}),
                "user_mod_file": (prompt_files, {"default": default_user_mod}),
                "max_tokens": ("INT", {"default": 4096, "min": 512, "max": 8192}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "include_system_prompt": ("BOOLEAN", {"default": True}),
                "include_think_block": ("BOOLEAN", {"default": False}),
                "strip_quotes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_analysis_prompt": ("STRING", {"multiline": True, "default": ""}),
                "user_modification": ("STRING", {"multiline": True, "default": ""}),
                "custom_system_prompt": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "zimage_formatted",
        "description",
        "raw_json",
    )
    FUNCTION = "analyze_image"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    @staticmethod
    def tensor2pil(image):
        """Convert tensor to PIL image"""
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @classmethod
    def ensure_model(cls, model_name):
        """Download model if not present"""
        models_dir = Path(folder_paths.models_dir) / "LLM" / "Qwen-VL"
        models_dir.mkdir(parents=True, exist_ok=True)

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
        if keep_loaded and cls._cached_model is not None and cls._cached_model_name == model_name:
            print(f"♻️ Using cached {model_name}")
            return cls._cached_model, cls._cached_processor, cls._cached_tokenizer

        if cls._cached_model_name != model_name:
            cls.clear_model()

        model_path = cls.ensure_model(model_name)
        print(f"🔄 Loading {model_name}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            device_map={"": 0} if device == "cuda" else device,
            dtype=dtype,
            attn_implementation="sdpa",
            use_safetensors=True,
        ).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

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

    def load_prompt_from_file(self, prompt_file):
        """Load prompt content from a file in the custom_prompts directory"""
        if prompt_file == "(none)" or not prompt_file:
            return None
        
        file_path = os.path.join(prompt_dir, prompt_file)
        if not os.path.exists(file_path):
            print(f"⚠️ Prompt file not found: {file_path}")
            return None
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return f.read().strip()
        except Exception as e:
            print(f"⚠️ Error loading prompt file: {e}")
            return None

    def build_analysis_prompt(self, prompt_file="(none)", custom_prompt=""):
        """Build the prompt for image analysis - priority: custom_prompt > prompt_file > default"""
        # Custom prompt string takes highest priority
        if custom_prompt.strip():
            return custom_prompt.strip()
        
        # Try to load from file
        file_prompt = self.load_prompt_from_file(prompt_file)
        if file_prompt:
            return file_prompt
        
        # Fallback to hardcoded default (should rarely be needed now)
        return """Analyze this image in detail and output a JSON object describing what you see. Include any relevant details about:

- Subject(s): People, characters, objects, animals - describe their appearance, clothing, expressions, poses
- Setting: Location, environment, background elements
- Style: Art style, photography style, lighting, color palette, mood
- Composition: Camera angle, framing, focal points
- Any text visible in the image
- Notable details or unique elements

Output ONLY valid JSON. Structure the JSON however makes sense for the image content. Be thorough and descriptive.

CRITICAL: Output ONLY the JSON object. No explanations, no markdown code blocks, no extra text. Start with { and end with }."""

    def json_to_description(self, data, indent=0):
        """Convert JSON data to a readable description string"""
        lines = []
        prefix = "  " * indent
        
        if isinstance(data, dict):
            for key, value in data.items():
                # Convert key from snake_case to Title Case
                readable_key = key.replace("_", " ").title()
                if isinstance(value, (dict, list)):
                    lines.append(f"{prefix}**{readable_key}:**")
                    lines.append(self.json_to_description(value, indent + 1))
                else:
                    lines.append(f"{prefix}- {readable_key}: {value}")
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(self.json_to_description(item, indent))
                    lines.append("")  # Add spacing between list items
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{data}")
        
        return "\n".join(lines)

    def format_zimage_output(self, content, include_system=True, include_think=True, 
                             custom_system="", user_mod=""):
        """Format content into Z-Image TurnBuilder chat format"""
        
        # Build Z-Image formatted output with chat tokens
        zimage_parts = []
        
        # System prompt
        if include_system:
            system_prompt = custom_system.strip() if custom_system.strip() else (
                "Generate an image based on the detailed visual specification provided. "
                "Follow the description precisely, paying attention to all visual details, "
                "composition, lighting, and style mentioned."
            )
            zimage_parts.append(f"<|im_start|>system\n{system_prompt}<|im_end|>")
        
        # User turn with content
        zimage_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
        
        # Assistant response
        if include_think:
            think_content = "Analyzing the visual specification. Processing key elements: subjects, setting, style, lighting, and composition details."
            zimage_parts.append(f"<|im_start|>assistant\n<think>\n{think_content}\n</think>\n\nHere is the generated image.<|im_end|>")
        else:
            zimage_parts.append(f"<|im_start|>assistant\n<|im_end|>")
        
        # Add user modification if provided
        if user_mod.strip():
            zimage_parts.append(f"<|im_start|>user\n{user_mod}<|im_end|>")
            if include_think:
                zimage_parts.append("<|im_start|>assistant\n<think>\nProcessing the modification request.\n</think>\n\nOk, here's the updated image.<|im_end|>")
            else:
                zimage_parts.append("<|im_start|>assistant\n<|im_end|>")
        
        # Final user/assistant turn for generation
        zimage_parts.append("<|im_start|>user\n<|im_end|>")
        zimage_parts.append("<|im_start|>assistant\n")
        
        zimage_formatted = "\n\n".join(zimage_parts)
        
        return zimage_formatted

    def analyze_image(self, images, qwen_model="Qwen3-VL-4B-Instruct", prompt_file="(none)",
                      system_prompt_file="(none)", user_mod_file="(none)",
                      max_tokens=4096, temperature=0.7, keep_model_loaded=True, 
                      include_system_prompt=True, include_think_block=True, strip_quotes=False, 
                      custom_analysis_prompt="", user_modification="", custom_system_prompt=""):
        try:
            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded)

            # Get first image
            if len(images.shape) == 3:
                pil_image = self.tensor2pil(images)
            else:
                pil_image = self.tensor2pil(images[0])

            # Build prompt (custom_analysis_prompt takes priority over prompt_file)
            prompt = self.build_analysis_prompt(prompt_file, custom_analysis_prompt)
            
            # Load system prompt (custom_system_prompt takes priority over system_prompt_file)
            system_prompt = custom_system_prompt.strip() if custom_system_prompt.strip() else (self.load_prompt_from_file(system_prompt_file) or "")
            
            # Load user modification (user_modification takes priority over user_mod_file)
            user_mod = user_modification.strip() if user_modification.strip() else (self.load_prompt_from_file(user_mod_file) or "")

            # Prepare conversation
            conversation = [{"role": "user", "content": []}]
            conversation[0]["content"].append({"type": "image", "image": pil_image})
            conversation[0]["content"].append({"type": "text", "text": prompt})

            # Apply chat template
            chat = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Process inputs
            processed = processor(text=chat, images=[pil_image], return_tensors="pt")

            # Move to device
            model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }

            print(f"🔄 Z-Image Vision: Analyzing image with {qwen_model}...")

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

            # Decode
            input_len = model_inputs["input_ids"].shape[-1]
            content = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            print(f"✅ Z-Image Vision: Received response ({len(content)} chars)")

            if not content:
                print("⚠️ Model returned empty response")
                if not keep_model_loaded:
                    self.clear_model()
                return ("Error: Empty response", "Error: Empty response", "{}")

            # Extract JSON from response
            json_str = content
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
                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1 and end > start:
                    json_str = content[start:end+1].strip()

            # Parse JSON
            data = json.loads(json_str)
            raw_json = json.dumps(data, indent=2)
            
            # Convert JSON to readable description
            description = self.json_to_description(data)

            # Format Z-Image output
            zimage_formatted = self.format_zimage_output(
                description,
                include_system=include_system_prompt,
                include_think=include_think_block,
                custom_system=system_prompt,
                user_mod=user_mod
            )

            # Strip quotes if requested
            if strip_quotes:
                zimage_formatted = zimage_formatted.replace('"', '')
                description = description.replace('"', '')
                raw_json = raw_json.replace('"', '')

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            print(f"✅ Z-Image Vision: Generated description and Z-Image format")

            return (zimage_formatted, description, raw_json)

        except json.JSONDecodeError as e:
            print(f"❌ Z-Image Vision: Failed to parse JSON: {e}")
            if 'content' in locals() and content:
                # Return raw content as fallback - still wrap in Z-Image format
                zimage_fallback = self.format_zimage_output(
                    content,
                    include_system=include_system_prompt,
                    include_think=include_think_block,
                    custom_system=system_prompt,
                    user_mod=user_mod
                )
                if strip_quotes:
                    zimage_fallback = zimage_fallback.replace('"', '')
                    content = content.replace('"', '')
                if not keep_model_loaded:
                    self.clear_model()
                return (zimage_fallback, content, json.dumps({"error": str(e), "raw": content[:1000]}, indent=2))
            
            if not keep_model_loaded:
                self.clear_model()
            return ("JSON parse error", "JSON parse error", "{}")

        except Exception as e:
            import traceback
            print(f"❌ Z-Image Vision: Error: {e}")
            print(traceback.format_exc())
            if not keep_model_loaded:
                self.clear_model()
            return (f"Error: {str(e)}", f"Error: {str(e)}", "{}")

