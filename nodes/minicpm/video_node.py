# MiniCPM-V Video Node
# Implements MiniCPM-V-4.5 for video understanding using 3D-resampler with temporal IDs

import os
import torch
import numpy as np
import math
from PIL import Image
from scipy.spatial import cKDTree
from ...utils.constants import CUSTOM_CATEGORY

# Lazy imports for heavy dependencies
transformers = None
decord = None

def lazy_import_dependencies():
    """Lazy import of heavy dependencies only when needed"""
    global transformers, decord
    if transformers is None:
        try:
            import transformers as tf
            transformers = tf
        except ImportError:
            raise ImportError("transformers library not found. Please install with: pip install transformers")
    
    if decord is None:
        try:
            import decord as dc
            decord = dc
        except ImportError:
            raise ImportError("decord library not found. Please install with: pip install decord")
    
    return transformers, decord


class MiniCPMVideoNode:
    """
    MiniCPM-V 4.5 Video Understanding Node
    
    Uses the MiniCPM-V-4.5 model from OpenBMB for high-FPS video understanding.
    The 3D-resampler compresses multiple frames into 64 tokens using temporal_ids,
    achieving 96x compression rate for efficient video processing.
    """
    
    # Class-level model cache to avoid reloading
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self):
        self.MAX_NUM_FRAMES = 180
        self.MAX_NUM_PACKING = 3
        self.TIME_SCALE = 0.1
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe the video in detail."
                }),
                "model_name": (
                    ["openbmb/MiniCPM-V-4_5", "openbmb/MiniCPM-o-2_6"],
                    {"default": "openbmb/MiniCPM-V-4_5"}
                ),
                "fps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "max_num_frames": ("INT", {
                    "default": 180,
                    "min": 10,
                    "max": 500,
                    "step": 10
                }),
                "max_num_packing": ("INT", {
                    "default": 3,
                    "min": 1,
                    "max": 6,
                    "step": 1
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False
                }),
                "use_image_id": ("BOOLEAN", {
                    "default": False
                }),
                "max_slice_nums": ("INT", {
                    "default": 1,
                    "min": 1,
                    "max": 9,
                    "step": 1
                }),
                "device": (
                    ["cuda", "cpu"],
                    {"default": "cuda"}
                ),
                "unload_after_inference": ("BOOLEAN", {
                    "default": False
                }),
            },
            "optional": {
                "force_packing": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 6,
                    "step": 1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "frame_info")
    FUNCTION = "analyze_video"
    CATEGORY = CUSTOM_CATEGORY
    
    def map_to_nearest_scale(self, values, scale):
        """Map frame timestamps to nearest scale values"""
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]
    
    def group_array(self, arr, size):
        """Group array elements into chunks of specified size"""
        return [arr[i:i+size] for i in range(0, len(arr), size)]
    
    def encode_video(self, video_path, choose_fps=3, force_packing=None):
        """
        Encode video frames with temporal IDs for 3D-resampler
        
        Returns:
            frames: List of PIL Images
            frame_ts_id_group: List of temporal ID groups for 3D packing
        """
        tf, dc = lazy_import_dependencies()
        
        def uniform_sample(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]
        
        vr = dc.VideoReader(video_path, ctx=dc.cpu(0))
        fps = vr.get_avg_fps()
        video_duration = len(vr) / fps
        
        if choose_fps * int(video_duration) <= self.MAX_NUM_FRAMES:
            packing_nums = 1
            choose_frames = round(min(choose_fps, round(fps)) * min(self.MAX_NUM_FRAMES, video_duration))
        else:
            packing_nums = math.ceil(video_duration * choose_fps / self.MAX_NUM_FRAMES)
            if packing_nums <= self.MAX_NUM_PACKING:
                choose_frames = round(video_duration * choose_fps)
            else:
                choose_frames = round(self.MAX_NUM_FRAMES * self.MAX_NUM_PACKING)
                packing_nums = self.MAX_NUM_PACKING
        
        frame_idx = [i for i in range(0, len(vr))]
        frame_idx = np.array(uniform_sample(frame_idx, choose_frames))
        
        if force_packing and force_packing > 0:
            packing_nums = min(force_packing, self.MAX_NUM_PACKING)
        
        print("\n" + "="*80)
        print(f"📹 VIDEO ENCODING: {os.path.basename(video_path)}")
        print("="*80)
        print(f"⏱️  Duration: {video_duration:.2f}s")
        print(f"🎞️  FPS: {fps:.2f}")
        print(f"📊 Total frames: {len(vr)}")
        print(f"✂️  Selected frames: {len(frame_idx)}")
        print(f"📦 Packing number: {packing_nums}")
        print(f"🗜️  Compression: {len(vr)/len(frame_idx)*packing_nums:.1f}x")
        print("-"*80)
        
        frames = vr.get_batch(frame_idx).asnumpy()
        
        frame_idx_ts = frame_idx / fps
        scale = np.arange(0, video_duration, self.TIME_SCALE)
        
        frame_ts_id = self.map_to_nearest_scale(frame_idx_ts, scale) / self.TIME_SCALE
        frame_ts_id = frame_ts_id.astype(np.int32)
        
        assert len(frames) == len(frame_ts_id), "Frame count mismatch with temporal IDs"
        
        frames = [Image.fromarray(v.astype('uint8')).convert('RGB') for v in frames]
        frame_ts_id_group = self.group_array(frame_ts_id, packing_nums)
        
        return frames, frame_ts_id_group
    
    def load_model(self, model_name, device):
        """Load or retrieve cached model and tokenizer"""
        tf, _ = lazy_import_dependencies()
        
        cache_key = f"{model_name}_{device}"
        
        # Check if model is already loaded
        if cache_key in self._model_cache:
            print(f"✅ Using cached model: {model_name}")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        
        print("\n" + "="*80)
        print(f"📥 LOADING MODEL: {model_name}")
        print("="*80)
        
        try:
            # Load model with optimizations
            print("🔧 Loading model with sdpa attention and bfloat16...")
            model = tf.AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',  # sdpa or flash_attention_2
                torch_dtype=torch.bfloat16
            )
            model = model.eval()
            
            if device == "cuda":
                if not torch.cuda.is_available():
                    print("⚠️  CUDA not available, falling back to CPU")
                    device = "cpu"
                else:
                    print(f"🎮 Moving model to CUDA")
                    model = model.cuda()
            else:
                print("💻 Using CPU")
            
            print("📝 Loading tokenizer...")
            tokenizer = tf.AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            # Cache the model and tokenizer
            self._model_cache[cache_key] = model
            self._tokenizer_cache[cache_key] = tokenizer
            
            print("✅ Model loaded successfully!")
            print("="*80 + "\n")
            
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def unload_model(self, model_name, device):
        """Unload model from memory"""
        cache_key = f"{model_name}_{device}"
        
        if cache_key in self._model_cache:
            print(f"🗑️  Unloading model: {model_name}")
            del self._model_cache[cache_key]
            del self._tokenizer_cache[cache_key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ Model unloaded from memory")
    
    def analyze_video(
        self,
        video_path,
        question="Describe the video",
        model_name="openbmb/MiniCPM-V-4_5",
        fps=5,
        max_num_frames=180,
        max_num_packing=3,
        enable_thinking=False,
        use_image_id=False,
        max_slice_nums=1,
        device="cuda",
        unload_after_inference=False,
        force_packing=0,
    ):
        """
        Analyze video using MiniCPM-V model
        
        Args:
            video_path: Path to video file
            question: Question to ask about the video
            model_name: Model to use
            fps: Frames per second to sample
            max_num_frames: Maximum frames after packing
            max_num_packing: Maximum packing number (1-6)
            enable_thinking: Enable thinking mode
            use_image_id: Use image IDs
            max_slice_nums: Maximum slice numbers
            device: Device to use (cuda/cpu)
            unload_after_inference: Whether to unload model after inference
            force_packing: Force specific packing number (0 = auto)
        
        Returns:
            response: Model's response
            frame_info: Information about frame processing
        """
        try:
            # Validate inputs
            if not video_path or not os.path.exists(video_path):
                error_msg = f"Video file not found: {video_path}"
                print(f"❌ {error_msg}")
                return (error_msg, "")
            
            # Update instance variables
            self.MAX_NUM_FRAMES = max_num_frames
            self.MAX_NUM_PACKING = max_num_packing
            
            # Load model and tokenizer
            model, tokenizer = self.load_model(model_name, device)
            
            # Encode video frames
            force_packing_val = force_packing if force_packing > 0 else None
            frames, frame_ts_id_group = self.encode_video(
                video_path,
                choose_fps=fps,
                force_packing=force_packing_val
            )
            
            # Prepare messages
            msgs = [
                {'role': 'user', 'content': frames + [question]},
            ]
            
            print("\n" + "="*80)
            print("🤖 RUNNING INFERENCE")
            print("="*80)
            print(f"❓ Question: {question}")
            print(f"🧠 Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
            print("-"*80)
            
            # Run inference
            answer = model.chat(
                msgs=msgs,
                tokenizer=tokenizer,
                use_image_id=use_image_id,
                max_slice_nums=max_slice_nums,
                temporal_ids=frame_ts_id_group,
                enable_thinking=enable_thinking,
                stream=False
            )
            
            print("✅ INFERENCE COMPLETE")
            print("="*80)
            print("📝 RESPONSE:")
            print("-"*80)
            print(answer)
            print("="*80 + "\n")
            
            # Prepare frame info
            frame_info = (
                f"Video: {os.path.basename(video_path)}\n"
                f"Frames processed: {len(frames)}\n"
                f"Packing groups: {len(frame_ts_id_group)}\n"
                f"Temporal groups: {frame_ts_id_group[:5]}..." if len(frame_ts_id_group) > 5 else str(frame_ts_id_group)
            )
            
            # Unload model if requested
            if unload_after_inference:
                self.unload_model(model_name, device)
            
            return (answer, frame_info)
            
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}\n\nPlease install required packages:\npip install transformers decord scipy"
            print(f"❌ {error_msg}")
            return (error_msg, "")
            
        except Exception as e:
            error_msg = f"Error analyzing video: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg, "")

