# QwenVL Video Analyzer Node

import os
import numpy as np
import torch
import gc
import subprocess
from pathlib import Path
from PIL import Image
from huggingface_hub import snapshot_download
from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer

from ...utils.constants import CUSTOM_CATEGORY, qwenvl_models
import folder_paths

# Lazy imports for video libraries
decord = None
cv2 = None

def lazy_import_video_libs():
    """Try to import video libraries, return what's available"""
    global decord, cv2

    libs = []

    # Try decord
    if decord is None:
        try:
            import decord as dc
            decord = dc
            libs.append("decord")
        except ImportError:
            pass

    # Try opencv
    if cv2 is None:
        try:
            import cv2 as cv
            cv2 = cv
            libs.append("opencv")
        except ImportError:
            pass

    return libs


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
                "video_section": (["start", "end"], {"default": "start", "tooltip": "Extract frames from start or end of video"}),
                "frame_window": ("INT", {"default": 60, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to consider from the selected section"}),
                "fps_extract": ("INT", {"default": 3, "min": 1, "max": 30, "step": 1, "tooltip": "Number of frames to extract from the frame window"}),
                "qwen_model": (qwenvl_models, {"default": qwenvl_models[0] if qwenvl_models else "Qwen3-VL-4B-Instruct", "tooltip": "Qwen-VL model to use for analysis"}),
                "max_tokens": ("INT", {"default": 1024, "min": 64, "max": 4096, "tooltip": "Maximum tokens to generate"}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.1, "max": 1.0, "tooltip": "Sampling temperature (lower = more focused)"}),
                "keep_model_loaded": ("BOOLEAN", {"default": True, "tooltip": "Keep model in memory for faster subsequent runs"}),
            },
            "optional": {
                "video": ("VIDEO,IMAGE", {"tooltip": "Connect VIDEO from LoadVideo node or IMAGE batch"}),
                "custom_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Custom analysis prompt (overrides default)"}),
            },
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        """Force re-execution when inputs change"""
        import hashlib
        m = hashlib.sha256()

        # Check video input
        video = kwargs.get('video', None)
        if video is not None:
            # Try to get a unique identifier for the video
            if torch.is_tensor(video):
                m.update(str(video.shape).encode())
            else:
                m.update(str(type(video)).encode())

        # Add other parameters that should trigger re-execution
        m.update(str(kwargs.get('video_section', 'start')).encode())
        m.update(str(kwargs.get('frame_window', 60)).encode())
        m.update(str(kwargs.get('fps_extract', 3)).encode())

        return m.hexdigest()

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

    def extract_frames_from_video_file(self, video_path, video_section, frame_window, fps_extract):
        """
        Extract frames from a video file using available libraries.

        Args:
            video_path: Path to video file
            video_section: "start" or "end" - where to extract from
            frame_window: Number of frames to consider from start/end
            fps_extract: Number of frames to extract from the window

        Returns:
            Tuple of (list of PIL images, tensor of extracted frames)
        """
        print(f"📹 Extracting frames from video file: {video_path}")

        # Check which libraries are available
        available_libs = lazy_import_video_libs()

        if not available_libs:
            print("⚠️ No video libraries found (decord, opencv), trying ffmpeg...")
            return self._extract_with_ffmpeg(video_path, video_section, frame_window, fps_extract)

        # Try decord first (preferred)
        if "decord" in available_libs:
            try:
                return self._extract_with_decord(video_path, video_section, frame_window, fps_extract)
            except Exception as e:
                print(f"⚠️ Decord extraction failed: {e}")
                if "opencv" in available_libs:
                    print("Falling back to OpenCV...")
                    return self._extract_with_opencv(video_path, video_section, frame_window, fps_extract)
                else:
                    print("Falling back to ffmpeg...")
                    return self._extract_with_ffmpeg(video_path, video_section, frame_window, fps_extract)

        # Try opencv
        elif "opencv" in available_libs:
            try:
                return self._extract_with_opencv(video_path, video_section, frame_window, fps_extract)
            except Exception as e:
                print(f"⚠️ OpenCV extraction failed: {e}")
                print("Falling back to ffmpeg...")
                return self._extract_with_ffmpeg(video_path, video_section, frame_window, fps_extract)

    def _extract_with_decord(self, video_path, video_section, frame_window, fps_extract):
        """Extract frames using decord library"""
        print("📚 Using decord for frame extraction")

        vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
        total_frames = len(vr)
        fps = vr.get_avg_fps()

        print(f"📹 Total video frames: {total_frames}, FPS: {fps:.2f}")

        # Determine which frames to extract
        if video_section == "end":
            start_idx = max(0, total_frames - frame_window)
            end_idx = total_frames
        else:  # "start"
            start_idx = 0
            end_idx = min(frame_window, total_frames)

        window_size = end_idx - start_idx

        # Calculate indices for fps_extract frames
        if window_size <= fps_extract:
            indices = list(range(start_idx, end_idx))
        else:
            indices = np.linspace(start_idx, end_idx - 1, fps_extract, dtype=int).tolist()

        print(f"📹 Extracting frames at indices: {indices}")

        # Extract frames
        frames_array = vr.get_batch(indices).asnumpy()
        frames_pil = [Image.fromarray(frame.astype('uint8')).convert('RGB') for frame in frames_array]

        # Convert to tensor format
        frames_tensor = torch.from_numpy(frames_array).float() / 255.0

        print(f"✅ Extracted {len(frames_pil)} frames using decord")
        return frames_pil, frames_tensor

    def _extract_with_opencv(self, video_path, video_section, frame_window, fps_extract):
        """Extract frames using opencv library"""
        print("📚 Using OpenCV for frame extraction")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"📹 Total video frames: {total_frames}, FPS: {fps:.2f}")

        # Determine which frames to extract
        if video_section == "end":
            start_idx = max(0, total_frames - frame_window)
            end_idx = total_frames
        else:  # "start"
            start_idx = 0
            end_idx = min(frame_window, total_frames)

        window_size = end_idx - start_idx

        # Calculate indices for fps_extract frames
        if window_size <= fps_extract:
            indices = list(range(start_idx, end_idx))
        else:
            indices = np.linspace(start_idx, end_idx - 1, fps_extract, dtype=int).tolist()

        print(f"📹 Extracting frames at indices: {indices}")

        # Extract frames
        frames_pil = []
        frames_array = []

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames_pil.append(Image.fromarray(frame_rgb))
                frames_array.append(frame_rgb)

        cap.release()

        # Convert to tensor format
        frames_tensor = torch.from_numpy(np.array(frames_array)).float() / 255.0

        print(f"✅ Extracted {len(frames_pil)} frames using OpenCV")
        return frames_pil, frames_tensor

    def _extract_with_ffmpeg(self, video_path, video_section, frame_window, fps_extract):
        """Extract frames using ffmpeg subprocess as fallback"""
        print("📚 Using ffmpeg for frame extraction")

        # Get video info using ffprobe
        try:
            probe_cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-count_packets',
                '-show_entries', 'stream=nb_read_packets,r_frame_rate',
                '-of', 'csv=p=0',
                video_path
            ]
            result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
            output = result.stdout.strip().split(',')
            total_frames = int(output[0])
            fps_str = output[1].split('/')
            fps = float(fps_str[0]) / float(fps_str[1]) if len(fps_str) == 2 else float(fps_str[0])

            print(f"📹 Total video frames: {total_frames}, FPS: {fps:.2f}")

        except Exception as e:
            print(f"⚠️ Could not get video info: {e}")
            # Fallback values
            total_frames = frame_window
            fps = 30

        # Determine which frames to extract
        if video_section == "end":
            start_idx = max(0, total_frames - frame_window)
            end_idx = total_frames
        else:  # "start"
            start_idx = 0
            end_idx = min(frame_window, total_frames)

        window_size = end_idx - start_idx

        # Calculate indices for fps_extract frames
        if window_size <= fps_extract:
            indices = list(range(start_idx, end_idx))
        else:
            indices = np.linspace(start_idx, end_idx - 1, fps_extract, dtype=int).tolist()

        print(f"📹 Extracting frames at indices: {indices}")

        # Extract frames using ffmpeg
        frames_pil = []
        frames_array = []

        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            for idx in indices:
                output_file = os.path.join(temp_dir, f"frame_{idx}.png")

                # Extract specific frame
                ffmpeg_cmd = [
                    'ffmpeg', '-y',
                    '-i', video_path,
                    '-vf', f'select=eq(n\\,{idx})',
                    '-vframes', '1',
                    output_file
                ]

                try:
                    subprocess.run(ffmpeg_cmd, capture_output=True, check=True)

                    if os.path.exists(output_file):
                        img = Image.open(output_file).convert('RGB')
                        frames_pil.append(img)
                        frames_array.append(np.array(img))
                except Exception as e:
                    print(f"⚠️ Failed to extract frame {idx}: {e}")

        if not frames_pil:
            raise RuntimeError("Failed to extract any frames with ffmpeg")

        # Convert to tensor format
        frames_tensor = torch.from_numpy(np.array(frames_array)).float() / 255.0

        print(f"✅ Extracted {len(frames_pil)} frames using ffmpeg")
        return frames_pil, frames_tensor

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
        video_section="start",
        frame_window=60,
        fps_extract=3,
        qwen_model="Qwen3-VL-4B-Instruct",
        max_tokens=1024,
        temperature=0.7,
        keep_model_loaded=True,
        video=None,
        custom_prompt="",
    ):
        try:
            # Determine video source and extract frames
            frames_pil = None
            frames_tensor = None

            if video is None:
                error_msg = "No video input provided. Please connect either a VIDEO or IMAGE input."
                print(f"❌ {error_msg}")
                return (error_msg, torch.zeros((1, 64, 64, 3)))

            # Try to extract frames from video input (handles both VIDEO and IMAGE types)
            video_tensor = None

            # Method 1: Check if it's already a torch tensor (IMAGE type)
            if torch.is_tensor(video):
                print("🖼️  Detected IMAGE tensor input")
                video_tensor = video

            # Method 2: Check if it's a VIDEO type with get_components method
            elif hasattr(video, 'get_components'):
                print("🎬 Detected VIDEO type with get_components")
                try:
                    components = video.get_components()
                    if hasattr(components, 'images'):
                        video_tensor = components.images
                    elif torch.is_tensor(components):
                        video_tensor = components
                    else:
                        # Try to access as dict or object
                        video_tensor = getattr(components, 'images', components)
                except Exception as e:
                    print(f"⚠️ Failed to get components: {e}, trying alternative methods...")

            # Method 3: Check if it has an images attribute directly
            if video_tensor is None and hasattr(video, 'images'):
                print("🎬 Detected VIDEO type with images attribute")
                video_tensor = video.images

            # Method 4: Check if video is a list of tensors or images
            if video_tensor is None and isinstance(video, (list, tuple)):
                print("🎬 Detected VIDEO as list/tuple")
                if len(video) > 0 and torch.is_tensor(video[0]):
                    video_tensor = torch.stack(video) if len(video) > 1 else video[0]

            # Method 5: Try calling it if it's callable
            if video_tensor is None and callable(video):
                print("🎬 Detected VIDEO as callable")
                try:
                    result = video()
                    if torch.is_tensor(result):
                        video_tensor = result
                except:
                    pass

            # If we still don't have a tensor, raise an error
            if video_tensor is None:
                error_msg = f"Could not extract frames from video input. Type: {type(video)}, Attributes: {dir(video)[:10]}"
                print(f"❌ {error_msg}")
                return (error_msg, torch.zeros((1, 64, 64, 3)))

            # Now extract frames from the tensor
            print(f"📹 Video tensor shape: {video_tensor.shape}")
            frames_pil, frames_tensor = self.extract_frames(
                video_tensor, video_section, frame_window, fps_extract
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

            # Return error and frames if available, otherwise dummy tensor
            if frames_tensor is not None:
                return (error_msg, frames_tensor)
            elif images is not None:
                return (error_msg, images)
            else:
                return (error_msg, torch.zeros((1, 64, 64, 3)))
