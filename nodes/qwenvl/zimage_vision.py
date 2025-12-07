# QwenVL Z-Image Vision Node
# Analyzes images and outputs in Z-Image TurnBuilder chat format

import json
import re
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
# huggingface_hub not needed - transformers handles downloads automatically
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
    _cached_attn_impl = None
    
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
                "temperature": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "use_flash_attention": ("BOOLEAN", {"default": False}),
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
    def load_model(cls, model_name, keep_loaded=True, use_flash_attention=False):
        """Load model with caching support"""
        attn_impl = "flash_attention_2" if use_flash_attention else "sdpa"
        
        # Check if we can use cached model (same model AND same attention implementation)
        if (keep_loaded and cls._cached_model is not None and 
            cls._cached_model_name == model_name and cls._cached_attn_impl == attn_impl):
            print(f"♻️ Using cached {model_name} (attn: {attn_impl})")
            return cls._cached_model, cls._cached_processor, cls._cached_tokenizer

        # Clear if model or attention implementation changed
        if cls._cached_model_name != model_name or cls._cached_attn_impl != attn_impl:
            cls.clear_model()

        model_path, cache_dir = cls.ensure_model(model_name)
        print(f"🔄 Loading {model_name} with {attn_impl}...")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if use_flash_attention and torch.cuda.is_available() else (torch.float16 if torch.cuda.is_available() else torch.float32)

        # Build kwargs - add cache_dir if downloading
        load_kwargs = {
            "device_map": {"": 0} if device == "cuda" else device,
            "dtype": dtype,
            "attn_implementation": attn_impl,
            "use_safetensors": True,
            "trust_remote_code": True,
        }
        if cache_dir:
            load_kwargs["cache_dir"] = cache_dir

        model = AutoModelForVision2Seq.from_pretrained(model_path, **load_kwargs).eval()

        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, cache_dir=cache_dir)

        if keep_loaded:
            cls._cached_model = model
            cls._cached_processor = processor
            cls._cached_tokenizer = tokenizer
            cls._cached_model_name = model_name
            cls._cached_attn_impl = attn_impl

        print(f"✅ Loaded {model_name} with {attn_impl}")
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
            cls._cached_attn_impl = None
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

    def json_to_zimage_prompt(self, data):
        """
        Convert JSON data to Z-Image prompt format.
        Creates a detailed, comma-separated prompt optimized for Z-Image generation.
        Format: Subject details, setting/environment, lighting/mood, style, quality boosters
        """
        segments = []
        
        # Priority order for building the prompt
        priority_keys = [
            # Subject-related
            'subject', 'subjects', 'character', 'characters', 'person', 'people', 
            'figure', 'main_subject', 'protagonist', 'portrait',
            # Appearance
            'appearance', 'description', 'look', 'features', 'face', 'facial_features',
            'hair', 'eyes', 'skin', 'body', 'clothing', 'outfit', 'attire', 'accessories',
            # Action/pose
            'action', 'pose', 'gesture', 'expression', 'emotion', 'activity',
            # Setting/environment
            'setting', 'environment', 'location', 'scene', 'background', 'backdrop',
            'place', 'surroundings', 'context',
            # Time/weather
            'time', 'time_of_day', 'weather', 'season', 'atmosphere',
            # Lighting
            'lighting', 'light', 'illumination', 'shadows',
            # Mood/tone
            'mood', 'tone', 'feeling', 'vibe', 'ambiance',
            # Style
            'style', 'art_style', 'aesthetic', 'genre', 'medium', 'technique',
            # Composition
            'composition', 'framing', 'camera', 'camera_angle', 'perspective', 'shot_type',
            'depth_of_field', 'focal_point',
            # Colors
            'colors', 'color_palette', 'palette', 'color_scheme',
            # Quality/technical
            'quality', 'resolution', 'detail', 'details', 'technical',
        ]
        
        def extract_value(val):
            """Recursively extract string values from nested structures"""
            if isinstance(val, str):
                return val.strip()
            elif isinstance(val, list):
                parts = []
                for item in val:
                    extracted = extract_value(item)
                    if extracted:
                        parts.append(extracted)
                return ", ".join(parts)
            elif isinstance(val, dict):
                parts = []
                for k, v in val.items():
                    extracted = extract_value(v)
                    if extracted:
                        # Include key context for nested dicts
                        key_readable = k.replace("_", " ")
                        parts.append(f"{key_readable}: {extracted}")
                return ", ".join(parts)
            elif val is not None:
                return str(val).strip()
            return ""
        
        if isinstance(data, dict):
            processed_keys = set()
            
            # First pass: process priority keys in order
            for key in priority_keys:
                for data_key in data.keys():
                    if data_key.lower() == key or key in data_key.lower():
                        if data_key not in processed_keys:
                            value = extract_value(data[data_key])
                            if value:
                                segments.append(value)
                            processed_keys.add(data_key)
            
            # Second pass: process any remaining keys
            for key, value in data.items():
                if key not in processed_keys:
                    extracted = extract_value(value)
                    if extracted:
                        segments.append(extracted)
        
        elif isinstance(data, list):
            for item in data:
                extracted = extract_value(item)
                if extracted:
                    segments.append(extracted)
        else:
            segments.append(str(data))
        
        # Join all segments with commas
        prompt = ", ".join(segments)
        
        # Clean up the prompt
        # Remove duplicate commas and extra spaces
        prompt = re.sub(r',\s*,', ',', prompt)
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = prompt.strip().strip(',').strip()
        
        # Add quality boosters if not already present
        quality_terms = ['8k', '4k', 'highly detailed', 'hyper-detailed', 'photorealistic', 
                        'ultra detailed', 'high resolution', 'hd', 'uhd']
        has_quality = any(term in prompt.lower() for term in quality_terms)
        
        if not has_quality and len(prompt) > 50:
            prompt += ", highly detailed, 8K"
        
        return prompt

    def analyze_image(self, images, qwen_model="Qwen3-VL-4B-Instruct", prompt_file="(none)",
                      system_prompt_file="(none)", user_mod_file="(none)",
                      max_tokens=4096, temperature=0.5, keep_model_loaded=True, use_flash_attention=False,
                      include_system_prompt=True, include_think_block=True, strip_quotes=False, 
                      custom_analysis_prompt="", user_modification="", custom_system_prompt=""):
        try:
            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded, use_flash_attention)

            # Convert all images to PIL format
            pil_images = []
            if len(images.shape) == 3:
                # Single image (H, W, C)
                pil_images.append(self.tensor2pil(images))
            else:
                # Multiple images (B, H, W, C)
                for i in range(images.shape[0]):
                    pil_images.append(self.tensor2pil(images[i]))
            
            num_images = len(pil_images)
            print(f"🖼️ Processing {num_images} image(s)")

            # Build prompt (custom_analysis_prompt takes priority over prompt_file)
            prompt = self.build_analysis_prompt(prompt_file, custom_analysis_prompt)
            print(f"📄 Using prompt file: {prompt_file}")

            # Prepare conversation with all images
            conversation = [{"role": "user", "content": []}]
            for pil_image in pil_images:
                conversation[0]["content"].append({"type": "image", "image": pil_image})
            conversation[0]["content"].append({"type": "text", "text": prompt})

            # Apply chat template
            chat = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Process inputs with all images
            processed = processor(text=chat, images=pil_images, return_tensors="pt")

            # Move to device
            model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }

            print(f"🔄 Z-Image Vision: Analyzing {num_images} image(s) with {qwen_model}...")

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
            print(f"📝 Raw response from Qwen model:\n{content}\n{'='*60}")

            if not content:
                print("⚠️ Model returned empty response")
                if not keep_model_loaded:
                    self.clear_model()
                return ("Error: Empty response", "Error: Empty response", "{}")

            # Try to parse as JSON first, otherwise use content as-is
            data = None
            
            # Method 1: Try direct JSON parse
            try:
                data = json.loads(content)
                print(f"✅ Parsed response as JSON directly")
            except json.JSONDecodeError:
                pass
            
            # Method 2: Try to extract JSON from content (code blocks or embedded)
            if data is None:
                json_str = None
                if "```json" in content:
                    start = content.find("```json") + 7
                    end = content.find("```", start)
                    if end != -1:
                        json_str = content[start:end].strip()
                elif "{" in content and "}" in content:
                    start = content.find("{")
                    end = content.rfind("}")
                    if start != -1 and end != -1 and end > start:
                        json_str = content[start:end+1].strip()
                
                if json_str:
                    try:
                        data = json.loads(json_str)
                        print(f"✅ Extracted and parsed JSON from response")
                    except json.JSONDecodeError:
                        pass
            
            # If JSON found, convert it
            if data is not None:
                raw_json = json.dumps(data, indent=2)
                description = self.json_to_description(data)
                zimage_formatted = self.json_to_zimage_prompt(data)
            else:
                # No JSON - use content directly as description/prompt
                print(f"ℹ️ Using raw content as description (no JSON)")
                clean_content = content.strip()
                # Remove thinking blocks if present
                if "<think>" in clean_content and "</think>" in clean_content:
                    think_end = clean_content.find("</think>") + 8
                    clean_content = clean_content[think_end:].strip()
                
                zimage_formatted = clean_content
                description = clean_content
                raw_json = json.dumps({"raw_response": content}, indent=2)

            # Strip quotes if requested
            if strip_quotes:
                zimage_formatted = zimage_formatted.replace('"', '')
                description = description.replace('"', '')
                raw_json = raw_json.replace('"', '')

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            print(f"✅ Z-Image Vision: Generated description and Z-Image prompt")

            return (zimage_formatted, description, raw_json)


        except Exception as e:
            import traceback
            print(f"❌ Z-Image Vision: Error: {e}")
            print(traceback.format_exc())
            if not keep_model_loaded:
                self.clear_model()
            return (f"Error: {str(e)}", f"Error: {str(e)}", "{}")

