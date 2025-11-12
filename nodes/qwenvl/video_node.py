# QwenVL Video Analyzer Node

import os
import numpy as np
import torch
import gc
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models
import folder_paths


class QwenVLVideoNode:
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
                "images": ("IMAGE", {"tooltip": "Video frames from LoadVideo node (IMAGE batch)"}),
                "video_section": (["start", "end"], {"default": "start", "tooltip": "Extract frames from start or end of video"}),
                "frame_window": ("INT", {"default": 60, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to consider from the selected section"}),
                "fps_extract": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1, "tooltip": "Number of frames to extract from the frame window"}),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct", "tooltip": "Qwen-VL model to use for analysis"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "tooltip": "Maximum tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "tooltip": "Sampling temperature (lower = more focused)"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep model in memory for faster subsequent runs"}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Custom analysis prompt (overrides default)"}),
            },
        }

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("analysis", "extracted_frames")
    FUNCTION = "analyze_video"
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

    def extract_frames(self, video_tensor, video_section, frame_window, fps_extract):
        """
        Extract frames from video based on parameters.

        Args:
            video_tensor: Tensor containing video frames [num_frames, height, width, channels]
            video_section: "start" or "end" - where to extract from
            frame_window: Number of frames to consider from start/end
            fps_extract: Number of frames to extract from the window

        Returns:
            Tuple of (list of PIL images, tensor of extracted frames)
        """
        total_frames = video_tensor.shape[0]

        print(f"📹 Total video frames: {total_frames}")
        print(f"📹 Extracting from: {video_section}, window: {frame_window}, fps: {fps_extract}")

        # Determine which frames to consider based on video_section
        if video_section == "end":
            # Take last frame_window frames
            start_idx = max(0, total_frames - frame_window)
            end_idx = total_frames
        else:  # "start"
            # Take first frame_window frames
            start_idx = 0
            end_idx = min(frame_window, total_frames)

        # Get the frame window
        frame_window_tensor = video_tensor[start_idx:end_idx]
        window_size = frame_window_tensor.shape[0]

        print(f"📹 Frame window: {start_idx} to {end_idx} ({window_size} frames)")

        # Extract fps_extract frames evenly from the window
        if window_size <= fps_extract:
            # If window is smaller than or equal to fps_extract, take all frames
            indices = list(range(window_size))
        else:
            # Evenly distribute fps_extract frames across the window
            indices = np.linspace(0, window_size - 1, fps_extract, dtype=int).tolist()

        print(f"📹 Extracting frames at indices: {indices}")

        # Extract frames
        extracted_frames_tensor = frame_window_tensor[indices]
        extracted_frames_pil = [self.tensor_to_pil(frame) for frame in extracted_frames_tensor]

        print(f"✅ Extracted {len(extracted_frames_pil)} frames")

        return extracted_frames_pil, extracted_frames_tensor

    def analyze_video(
        self,
        images,
        video_section="start",
        frame_window=60,
        fps_extract=3,
        qwen_model="Qwen3-VL-4B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        keep_model_loaded=True,
        custom_prompt="",
    ):
        try:
            # Extract frames from video
            frames_pil, frames_tensor = self.extract_frames(
                images, video_section, frame_window, fps_extract
            )

            # Build prompt
            if custom_prompt.strip():
                prompt_text = custom_prompt.strip()
            else:
                prompt_text = f"""Analyze this video sequence and provide a detailed description. The video shows {len(frames_pil)} frames extracted from the {"beginning" if video_section == "start" else "end"} of the video.

Describe:
1. What is happening in the video sequence
2. Any movement or changes between frames
3. The overall scene, setting, and atmosphere
4. Key objects, people, or elements visible
5. The story or narrative being conveyed
6. Any text or important visual details

Provide a comprehensive analysis that captures the essence of this video segment."""

            # Load model
            model, processor, tokenizer = self.load_model(qwen_model, keep_model_loaded)

            # Prepare conversation with video frames
            conversation = [{"role": "user", "content": []}]

            # Add video frames
            conversation[0]["content"].append({"type": "video", "video": frames_pil})

            # Add text prompt
            conversation[0]["content"].append({"type": "text", "text": prompt_text})

            # Apply chat template
            chat = processor.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )

            # Process inputs - QwenVL expects videos as a list of frames
            processed = processor(text=chat, images=None, videos=[frames_pil], return_tensors="pt")

            # Move to device
            model_device = next(model.parameters()).device
            model_inputs = {
                key: value.to(model_device) if torch.is_tensor(value) else value
                for key, value in processed.items()
            }

            # Generate
            print(f"🔄 QwenVL Video Analyzer ({qwen_model}): Analyzing {len(frames_pil)} frames...")

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

            print(f"✅ QwenVL Video Analyzer ({qwen_model}): Generated {len(result)} characters")
            if len(result) < 200:
                print(f"📝 Full response: {result}")
            else:
                print(f"📝 First 200 chars: {result[:200]}")

            # Clear model if not keeping loaded
            if not keep_model_loaded:
                self.clear_model()

            return (result, frames_tensor)

        except Exception as e:
            import traceback
            error_msg = f"QwenVL Video Analyzer Error: {str(e)}"
            print(f"❌ {error_msg}")
            print(traceback.format_exc())

            # Clear model on error
            if not keep_model_loaded:
                self.clear_model()

            # Return error and original video frames
            return (error_msg, images)
