# QwenVL Next Scene Node
# Generates cinematic scene transitions using local QwenVL models

import os
import re
import random
import json
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models, prompt_dir
import folder_paths


class QwenVLNextScene:
    """
    Generates the next scene in a visual narrative using local QwenVL models.
    Takes an input prompt (previous scene description) and multiple frame images, then creates 
    a cinematic transition FROM the previous scene TO the next scene with camera movements, 
    framing evolution, and atmospheric shifts.
    
    Supports 1-5 frames as input for better motion/progression understanding.
    Supports custom prompt files from data/custom_prompts/ directory.
    """
    
    # Class-level cache for model
    _cached_model = None
    _cached_processor = None
    _cached_tokenizer = None
    _cached_model_name = None
    _cached_attn_impl = None
    
    # Default prompt file
    DEFAULT_PROMPT_FILE = "next_scene.txt"

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

    @classmethod
    def INPUT_TYPES(cls):
        prompt_files = cls.get_prompt_files()
        default_prompt = cls.DEFAULT_PROMPT_FILE if cls.DEFAULT_PROMPT_FILE in prompt_files else "(none)"
        
        return {
            "required": {
                "images": ("IMAGE",),
                "original_prompt": ("STRING", {"multiline": True, "default": ""}),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct"}),
                "prompt_file": (prompt_files, {"default": default_prompt}),
                "max_frames": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
                "add_scene_prefix": ("BOOLEAN", {"default": True}),
                "keep_model_loaded": ("BOOLEAN", {"default": True}),
                "use_flash_attention": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "scene_prefix_text": ("STRING", {"default": "NEW SCENE:"}),
                "focus_on": (
                    ["Automatic", "Camera Movement", "Framing Evolution", "Environmental Reveals", "Atmospheric Shifts"],
                    {"default": "Automatic"}
                ),
                "transition_intensity": (
                    ["Subtle", "Moderate", "Dramatic"],
                    {"default": "Moderate"}
                ),
                "max_tokens": ("INT", {"default": 1024, "min": 256, "max": 4096}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.5}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("next_scene_prompt", "short_description")
    FUNCTION = "generate_next_scene"
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
        
        # Check if model is already loaded with same attention implementation
        if (keep_loaded and cls._cached_model is not None and 
            cls._cached_model_name == model_name and cls._cached_attn_impl == attn_impl):
            print(f"♻️ Using cached {model_name} (attn: {attn_impl})")
            return cls._cached_model, cls._cached_processor, cls._cached_tokenizer

        # Clear old model if loading a different one or different attention
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

    def build_prompt(self, original_prompt, focus_on, transition_intensity, num_frames=1, prompt_file="next_scene.txt", custom_prompt=""):
        """Build the full prompt for scene generation"""
        
        # Priority: custom_prompt > prompt_file > default fallback
        system_prompt = None
        
        if custom_prompt and custom_prompt.strip():
            system_prompt = custom_prompt.strip()
            print(f"📄 Using custom prompt (inline)")
        else:
            system_prompt = self.load_prompt_from_file(prompt_file)
            if system_prompt:
                print(f"📄 Using prompt file: {prompt_file}")
        
        # Add multi-frame context if applicable
        frame_context = ""
        if num_frames > 1:
            frame_context = f"""
NOTE: You are provided with {num_frames} sequential frames showing the progression of the current scene.
Analyze the MOTION and CHANGES across these frames to understand the direction/flow of the scene.
Frame 1 is the earliest, Frame {num_frames} is the most recent.
Use this temporal information to predict a natural continuation.

"""
        
        if original_prompt and original_prompt.strip() and system_prompt:
            # Use the loaded system prompt with original prompt context for proper transitions
            full_prompt = system_prompt.replace("##ORIGINAL_PROMPT##", original_prompt)
            if frame_context:
                # Insert frame context after the original prompt section
                full_prompt = full_prompt.replace("Your task is to:", f"{frame_context}Your task is to:")
        else:
            # Create a simplified prompt that just analyzes the image without previous context
            full_prompt = f"""You are an expert cinematographer and visual storytelling assistant. 

Analyze the provided {"images" if num_frames > 1 else "image"} and generate a detailed description of the "Next Scene" in a cinematic narrative.
{frame_context}
NOTE: No previous scene prompt was provided, so focus on analyzing the current {"frames" if num_frames > 1 else "image"} and suggesting a natural cinematic evolution from what you see.

First, understand what's happening in the current {"sequence" if num_frames > 1 else "image"}, then create a natural, cinematic transition to the next scene.

Consider these elements:

CAMERA MOVEMENTS:
- Dolly shots (push-in, pull-back)
- Tracking moves (lateral, forward, backward)
- Panning (left, right)
- Tilting (up, down)
- Crane shots (rising, descending)
- Zooming (in, out)

FRAMING EVOLUTION:
- Wide shot to close-up transitions
- Close-up to wide shot reveals
- Angle shifts (high angle, low angle, dutch angle)
- Reframing of subjects

ENVIRONMENTAL REVEALS:
- New characters entering frame
- Expanded scenery and locations
- Background elements becoming foreground
- Hidden details revealed

ATMOSPHERIC SHIFTS:
- Lighting changes (dramatic to soft, bright to dark)
- Weather evolution (clear to storm, fog rolling in)
- Time-of-day transitions (day to night, golden hour)
- Color palette shifts
- Mood and tone evolution

OUTPUT FORMAT:
Start with "Next Scene:" and provide a detailed, cinematic description (2-4 sentences) that:
1. {"Continues the motion/direction observed in the frames" if num_frames > 1 else "Describes a specific camera movement or framing change"}
2. Mentions what new elements appear or what exits frame
3. Captures the atmospheric or environmental evolution
4. Uses precise, professional cinematography language

CRITICAL: OUTPUT ONLY THE FINISHED "NEXT SCENE:" PROMPT - NO explanations or preambles."""
        
        # Add focus guidance if specified
        if focus_on != "Automatic":
            full_prompt += f"\n\nFOCUS: Emphasize {focus_on.lower()} in your next scene description."
        
        # Add intensity guidance
        intensity_map = {
            "Subtle": "The transition should be subtle and gentle, maintaining most of the original composition.",
            "Moderate": "The transition should be noticeable but natural, introducing clear changes while maintaining continuity.",
            "Dramatic": "The transition should be bold and impactful, creating significant visual change while respecting the narrative."
        }
        full_prompt += f"\n\nINTENSITY: {intensity_map[transition_intensity]}"
        
        return full_prompt

    def extract_frames(self, images, max_frames):
        """Extract evenly spaced frames from an image batch"""
        pil_images = []
        
        # Handle different tensor shapes
        if len(images.shape) == 3:
            # Single image (H, W, C)
            pil_images.append(self.tensor2pil(images))
        elif len(images.shape) == 4:
            # Batch of images (B, H, W, C)
            batch_size = images.shape[0]
            
            if batch_size <= max_frames:
                # Use all available frames
                for i in range(batch_size):
                    pil_images.append(self.tensor2pil(images[i]))
            else:
                # Sample evenly spaced frames
                indices = np.linspace(0, batch_size - 1, max_frames, dtype=int)
                for idx in indices:
                    pil_images.append(self.tensor2pil(images[idx]))
        
        return pil_images

    def generate_next_scene(
        self,
        images,
        original_prompt="",
        qwen_model="Qwen3-VL-4B-Instruct",
        prompt_file="next_scene.txt",
        max_frames=3,
        seed=-1,
        randomize_each_run=True,
        add_scene_prefix=True,
        keep_model_loaded=True,
        use_flash_attention=False,
        custom_prompt="",
        scene_prefix_text="NEW SCENE:",
        focus_on="Automatic",
        transition_intensity="Moderate",
        max_tokens=1024,
        temperature=0.7
    ):
        try:
            # Handle seed for randomization
            if randomize_each_run and seed == -1:
                current_seed = random.randint(0, 0xffffffffffffffff)
            elif seed == -1:
                current_seed = 12345
            else:
                current_seed = seed
            
            random.seed(current_seed)
            
            print("\n" + "="*80)
            print("🎬 QWENVL NEXT SCENE - Processing")
            print("="*80)
            if original_prompt and original_prompt.strip():
                print(f"📝 Previous Scene (transitioning FROM): {original_prompt[:100]}{'...' if len(original_prompt) > 100 else ''}")
                print("✨ Using full transition mode - creating next scene based on previous prompt")
            else:
                print("📝 Previous Scene: (None - analyzing images only)")
                print("ℹ️  Using image-only mode - no previous prompt for transition")
            print(f"🤖 Model: {qwen_model}")
            print(f"🎯 Focus: {focus_on}")
            print(f"⚡ Intensity: {transition_intensity}")
            print(f"🎲 Using seed: {current_seed}")
            print("-"*80)
            
            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded, use_flash_attention)
            
            # Extract frames from the batch
            pil_images = self.extract_frames(images, max_frames)
            num_frames = len(pil_images)
            
            print(f"🖼️  Using {num_frames} frame{'s' if num_frames > 1 else ''}")
            for i, img in enumerate(pil_images):
                print(f"   Frame {i+1}: {img.size}")
            print("🔄 Generating next scene...")

            # Build the prompt with frame count context
            full_prompt = self.build_prompt(original_prompt, focus_on, transition_intensity, num_frames, prompt_file, custom_prompt)

            # Prepare conversation with multiple images
            conversation = [{"role": "user", "content": []}]
            
            # Add frame labels for multiple frames
            if num_frames > 1:
                for i, pil_image in enumerate(pil_images):
                    conversation[0]["content"].append({"type": "text", "text": f"Frame {i+1}:"})
                    conversation[0]["content"].append({"type": "image", "image": pil_image})
                conversation[0]["content"].append({"type": "text", "text": f"\n{full_prompt}"})
            else:
                # Single image - no frame label needed
                conversation[0]["content"].append({"type": "image", "image": pil_images[0]})
                conversation[0]["content"].append({"type": "text", "text": full_prompt})

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
            result = tokenizer.decode(outputs[0, input_len:], skip_special_tokens=True).strip()

            print(f"✅ Response received ({len(result)} chars)")
            
            if not result:
                print("⚠️ Model returned empty response")
                fallback_text = "The camera pulls back to reveal more of the surrounding environment, as lighting shifts to create a different mood and atmosphere."
                result = f"{scene_prefix_text} {fallback_text}" if add_scene_prefix else fallback_text
                short_description = result
            else:
                # Extract a shorter version (first sentence or first 150 chars)
                sentences = re.split(r'(?<=[.!?])\s+', result)
                short_description = sentences[0] if sentences else result[:150]

                print("-"*80)
                print("🎬 NEXT SCENE PROMPT (Full):")
                print("-"*80)
                print(result)
                print("-"*80)
                print("📋 Short Description:")
                print(short_description)
                print("="*80 + "\n")
            
            # Handle the scene prefix based on user preference
            if not add_scene_prefix:
                # Remove any existing "Next Scene:" or "NEW SCENE:" or custom prefix
                result = re.sub(r'^(Next Scene:|NEW SCENE:)\s*', '', result, flags=re.IGNORECASE).strip()
                short_description = re.sub(r'^(Next Scene:|NEW SCENE:)\s*', '', short_description, flags=re.IGNORECASE).strip()
            else:
                # Ensure the custom prefix is used
                # First, remove any existing standard prefixes
                result = re.sub(r'^(Next Scene:|NEW SCENE:)\s*', '', result, flags=re.IGNORECASE).strip()
                short_description = re.sub(r'^(Next Scene:|NEW SCENE:)\s*', '', short_description, flags=re.IGNORECASE).strip()
                
                # Then add the custom prefix (ensure it has proper spacing)
                prefix = scene_prefix_text.strip()
                if prefix and not prefix.endswith(' ') and not prefix.endswith(':'):
                    prefix += ' '
                elif prefix and prefix.endswith(':'):
                    prefix += ' '
                    
                result = prefix + result
                short_description = prefix + short_description

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            return (result, short_description)

        except Exception as e:
            import traceback
            print(f"\n❌ An error occurred in QwenVLNextScene: {e}")
            print(traceback.format_exc())
            print("="*80 + "\n")
            
            if not keep_model_loaded:
                self.clear_model()
                
            error_message = f"Error generating next scene: {str(e)}"
            return (error_message, error_message)
