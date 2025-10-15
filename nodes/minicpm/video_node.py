# MiniCPM-V Video Node
# Implements MiniCPM-V-4.5 for video understanding using 3D-resampler with temporal IDs

import os
import torch
import numpy as np
import math
from PIL import Image
from scipy.spatial import cKDTree
from ...utils.constants import CUSTOM_CATEGORY

# Import ComfyUI folder_paths for file browser
try:
    import folder_paths
except ImportError:
    folder_paths = None

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
        self.MAX_NUM_FRAMES = 32
        self.MAX_NUM_PACKING = 6
        self.TIME_SCALE = 0.1
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe the video in detail."
                }),
                "model_name": ([
                    "openbmb/MiniCPM-V-4_5",
                    "openbmb/MiniCPM-o-2_6",
                ], {
                    "default": "openbmb/MiniCPM-V-4_5",
                }),
                "precision": ([
                    "bfloat16",
                    "float16",
                ], {
                    "default": "bfloat16",
                    "tooltip": "float16 uses slightly less memory. bfloat16 is more stable."
                }),
                "fps": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 30,
                    "step": 1
                }),
                "max_num_frames": ("INT", {
                    "default": 180,
                    "min": 1,
                    "max": 1000,
                    "step": 1
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
                "use_last_frames": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If True, use the last X frames. If False, use the first X frames."
                }),
            },
            "optional": {
                "video": ("VIDEO,IMAGE",),  # Accept VIDEO from LoadVideo or IMAGE frames
                "video_from_input": ("STRING", {
                    "default": "",
                    "video_upload": True,  # Enables file browser button
                    "multiline": False
                }),
                "video_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "forceInput": True  # Allow connections from other nodes
                }),
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
    
    @staticmethod
    def tensor2pil(image):
        """Convert ComfyUI tensor to PIL Image"""
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    
    def map_to_nearest_scale(self, values, scale):
        """Map frame timestamps to nearest scale values"""
        tree = cKDTree(np.asarray(scale)[:, None])
        _, indices = tree.query(np.asarray(values)[:, None])
        return np.asarray(scale)[indices]
    
    def group_array(self, arr, size):
        """Group array elements into chunks of specified size"""
        return [arr[i:i+size] for i in range(0, len(arr), size)]
    
    def encode_video(self, video_path, choose_fps=3, force_packing=None, use_last_frames=False):
        """
        Encode video frames with temporal IDs for 3D-resampler
        
        Returns:
            frames: List of PIL Images
            frame_ts_id_group: List of temporal ID groups for 3D packing
        """
        tf, dc = lazy_import_dependencies()
        
        def uniform_sample(l, n, use_last=False):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            if use_last:
                # Sample uniformly but from the end
                start_offset = len(l) - n * gap
                idxs = [int(start_offset + i * gap + gap / 2) for i in range(n)]
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
        
        if use_last_frames:
            # Take frames from the end of the video
            frame_idx = np.array(uniform_sample(frame_idx, choose_frames, use_last=True))
            print(f"📍 Sampling from END of video (last frames)")
        else:
            # Take frames from throughout the video (default uniform sampling)
            frame_idx = np.array(uniform_sample(frame_idx, choose_frames, use_last=False))
            print(f"📍 Sampling from START/THROUGHOUT video")
        
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
    
    def load_model(self, model_name, device, precision="bfloat16"):
        """Load or retrieve cached model and tokenizer"""
        tf, _ = lazy_import_dependencies()
        
        cache_key = f"{model_name}_{device}_{precision}"
        
        # Check if model is already loaded
        if cache_key in self._model_cache:
            print(f"✅ Using cached model: {model_name} ({precision})")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        
        print("\n" + "="*80)
        print(f"📥 LOADING MODEL: {model_name}")
        print(f"⚙️  Precision: {precision}")
        print("="*80)
        print("⏳ This may take a while on first download...")
        print("💡 Progress will be shown below")
        
        try:
            # Determine torch dtype
            if precision == "float16":
                torch_dtype = torch.float16
                print("💾 Using FP16 precision (slightly less memory)")
            else:
                torch_dtype = torch.bfloat16
                print("💾 Using BF16 precision (more stable)")
            
            # Standard transformers model
            print("\n" + "-"*80)
            print("📥 DOWNLOADING MODEL FILES")
            print("-"*80)
            print("💡 Transformers will show download progress automatically")
            print("-"*80 + "\n")
            
            # Load model with optimizations
            model = tf.AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                attn_implementation='sdpa',
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
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
    
    def unload_model(self, model_name, device, precision="bfloat16"):
        """Unload model from GPU VRAM only (does NOT delete model files)"""
        cache_key = f"{model_name}_{device}_{precision}"
        
        if cache_key in self._model_cache:
            print(f"🔓 Unloading model from VRAM: {model_name}")
            print("💡 Note: Model files are NOT deleted, only unloaded from memory")
            
            # Move model to CPU before deleting (helps ensure VRAM is freed)
            if device == "cuda" and torch.cuda.is_available():
                try:
                    self._model_cache[cache_key] = self._model_cache[cache_key].cpu()
                except:
                    pass
            
            # Remove from cache
            del self._model_cache[cache_key]
            del self._tokenizer_cache[cache_key]
            
            # Clear CUDA cache to free VRAM
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print(f"✅ Model unloaded from GPU VRAM (files remain on disk)")
            else:
                print(f"✅ Model unloaded from memory")
    
    def analyze_video(
        self,
        question="Describe the video",
        model_name="openbmb/MiniCPM-V-4_5",
        precision="bfloat16",
        fps=5,
        max_num_frames=180,
        max_num_packing=3,
        enable_thinking=False,
        use_image_id=False,
        max_slice_nums=1,
        device="cuda",
        unload_after_inference=False,
        use_last_frames=False,
        video=None,
        video_from_input="",
        video_path="",
        force_packing=0,
    ):
        """
        Analyze video using MiniCPM-V model
        
        Args:
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
            video: Pre-loaded video frames (IMAGE tensor from another node)
            video_from_input: Video filename from ComfyUI/input/ (optional)
            video_path: Full path to video file from anywhere (optional)
            force_packing: Force specific packing number (0 = auto)
        
        Returns:
            response: Model's response
            frame_info: Information about frame processing
        """
        try:
            # Update instance variables
            self.MAX_NUM_FRAMES = max_num_frames
            self.MAX_NUM_PACKING = max_num_packing
            
            # Load model and tokenizer first
            model, tokenizer = self.load_model(model_name, device, precision)
            
            # Determine which video source to use
            frames = None
            frame_ts_id_group = None
            video_source_info = ""
            
            # Priority 1: Pre-loaded video from another node
            if video is not None:
                print("📹 Using pre-loaded video from input")
                
                # Check if it's a VIDEO type (has get_components method) or IMAGE tensor
                if hasattr(video, 'get_components'):
                    # VIDEO type from LoadVideo or CreateVideo nodes
                    print("🎬 Detected VIDEO type, extracting components...")
                    components = video.get_components()
                    video_images = components.images
                    
                    # video_images should be an ImageInput, convert to tensor if needed
                    if hasattr(video_images, 'shape'):
                        # Already a tensor
                        video_tensor = video_images
                    else:
                        # Try to get tensor representation
                        video_tensor = video_images
                    
                    # Convert to PIL frames
                    if len(video_tensor.shape) == 4:
                        pil_frames = [self.tensor2pil(frame) for frame in video_tensor]
                    else:
                        pil_frames = [self.tensor2pil(video_tensor)]
                    
                    num_frames = len(pil_frames)
                    print(f"📊 Extracted {num_frames} frames from VIDEO")
                else:
                    # IMAGE tensor type
                    print("🖼️  Detected IMAGE tensor type")
                    if len(video.shape) == 4:
                        # Batch of frames: (batch, height, width, channels)
                        pil_frames = [self.tensor2pil(frame) for frame in video]
                    else:
                        # Single frame
                        pil_frames = [self.tensor2pil(video)]
                    
                num_frames = len(pil_frames)
                print(f"📊 Received {num_frames} frames")
                
                # Sample frames if needed
                if num_frames > max_num_frames:
                    if use_last_frames:
                        print(f"⚠️  Using LAST {max_num_frames} frames from {num_frames} total frames")
                        # Take the last max_num_frames
                        pil_frames = pil_frames[-max_num_frames:]
                    else:
                        print(f"⚠️  Using FIRST {max_num_frames} frames from {num_frames} total frames")
                        # Take the first max_num_frames
                        pil_frames = pil_frames[:max_num_frames]
                
                # Generate temporal IDs for the frames
                force_packing_val = force_packing if force_packing > 0 else None
                if force_packing_val:
                    packing_nums = min(force_packing_val, self.MAX_NUM_PACKING)
                else:
                    # Auto-determine packing
                    if len(pil_frames) <= self.MAX_NUM_FRAMES:
                        packing_nums = 1
                    else:
                        packing_nums = min(
                            math.ceil(len(pil_frames) / self.MAX_NUM_FRAMES),
                            self.MAX_NUM_PACKING
                        )
                
                # Create temporal IDs (sequential for pre-loaded frames)
                frame_ts_ids = np.arange(len(pil_frames), dtype=np.int32)
                frame_ts_id_group = self.group_array(frame_ts_ids, packing_nums)
                
                frames = pil_frames
                video_source_info = f"Pre-loaded frames: {num_frames}"
                
                print(f"📦 Packing number: {packing_nums}")
                print(f"✂️  Final frame count: {len(frames)}")
                
            else:
                # Priority 2: File path sources
                final_video_path = None
                
                # Check video_path first
                if video_path and video_path.strip():
                    final_video_path = video_path.strip()
                    print(f"📂 Using video_path: {final_video_path}")
                
                # Check video_from_input second
                elif video_from_input and video_from_input.strip():
                    video_input = video_from_input.strip()
                    
                    # Check if it's a full path (contains drive letter, forward/backward slashes, or starts with /)
                    is_full_path = (
                        ':' in video_input or  # Windows: C:\path
                        video_input.startswith('\\\\') or  # UNC: \\server\path
                        video_input.startswith('/') or  # Unix: /path/to/file
                        '/' in video_input or  # Contains path separator
                        '\\' in video_input  # Contains Windows separator
                    )
                    
                    if is_full_path:
                        # Treat as absolute path
                        final_video_path = video_input
                        print(f"📂 Using full path from video_from_input: {final_video_path}")
                    else:
                        # Treat as filename in ComfyUI input directory
                        if folder_paths is not None:
                            try:
                                final_video_path = folder_paths.get_annotated_filepath(video_input)
                                print(f"📂 Using file from input directory: {video_input}")
                            except:
                                # If that fails, try as direct path
                                final_video_path = video_input
                                print(f"📂 Using direct path: {final_video_path}")
                        else:
                            final_video_path = video_input
                            print(f"📂 Using path: {final_video_path}")
            
                # Validate inputs
                if not final_video_path:
                    error_msg = "No video source provided. Use 'video' input, 'video_from_input', or 'video_path'."
                    print(f"❌ {error_msg}")
                    return (error_msg, "")
                
                if not os.path.exists(final_video_path):
                    error_msg = f"Video file not found: {final_video_path}"
                    print(f"❌ {error_msg}")
                    return (error_msg, "")
                
                # Encode video frames from file
                force_packing_val = force_packing if force_packing > 0 else None
                frames, frame_ts_id_group = self.encode_video(
                    final_video_path,
                    choose_fps=fps,
                    force_packing=force_packing_val,
                    use_last_frames=use_last_frames
                )
                video_source_info = f"File: {os.path.basename(final_video_path)}"
            
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
            
            # Run inference with transformers (works with both quantized and non-quantized)
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
                f"Source: {video_source_info}\n"
                f"Frames processed: {len(frames)}\n"
                f"Packing groups: {len(frame_ts_id_group)}\n"
                f"Temporal groups: {frame_ts_id_group[:5]}..." if len(frame_ts_id_group) > 5 else str(frame_ts_id_group)
            )
            
            # Unload model if requested
            if unload_after_inference:
                self.unload_model(model_name, device, precision)
            
            return (answer, frame_info)
            
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}\n\nPlease install required packages:\npip install transformers decord scipy"
            print(f"❌ {error_msg}")
            return (error_msg, "")
            
        except torch.cuda.OutOfMemoryError as e:
            error_msg = "CUDA Out of Memory Error!"
            print(f"\n❌ {error_msg}")
            print("="*80)
            print("💡 SOLUTIONS TO FIX OUT OF MEMORY:")
            print("="*80)
            print("1. ⭐ USE QUANTIZATION: Set quantization to '4-bit' (BEST SOLUTION!)")
            print("2. ✅ REDUCE max_num_frames: Try 16-32 instead of 180")
            print("3. ✅ INCREASE max_num_packing: Try 6 instead of 3")
            print("4. ✅ ENABLE unload_after_inference: Free memory after use")
            print("5. ✅ USE CPU device: Slower but won't run out of memory")
            print("6. ✅ Close other GPU applications")
            print("7. ✅ Process fewer frames: Lower fps parameter")
            print("="*80)
            print(f"\n📊 Current settings:")
            print(f"   - precision: {precision}")
            print(f"   - max_num_frames: {max_num_frames} (try 16-32 for less memory!)")
            print(f"   - max_num_packing: {max_num_packing} (try 6 for maximum compression!)")
            print(f"   - device: {device}")
            print("\n💡 BEST FIX: Reduce max_num_frames to 16-25!")
            print("="*80 + "\n")
            
            # Try to free memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("🗑️  Cleared CUDA cache")
            
            return (error_msg + " - See console for solutions", "Out of Memory")
            
        except Exception as e:
            error_msg = f"Error analyzing video: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg, "")
    
    @classmethod
    def IS_CHANGED(s, video=None, video_from_input="", video_path="", **kwargs):
        """Check if video source has changed"""
        import hashlib
        
        # If video tensor is provided, use its hash
        if video is not None:
            try:
                # Hash the tensor data
                video_bytes = video.cpu().numpy().tobytes()
                m = hashlib.sha256()
                m.update(video_bytes[:1024*1024])  # First 1MB
                return m.digest().hex()
            except:
                return float("NaN")
        
        # Otherwise check file paths
        final_path = None
        if video_path and video_path.strip():
            final_path = video_path.strip()
        elif video_from_input and video_from_input.strip():
            video_input = video_from_input.strip()
            
            # Check if it's a full path
            is_full_path = (
                ':' in video_input or
                video_input.startswith('\\\\') or
                video_input.startswith('/') or
                '/' in video_input or
                '\\' in video_input
            )
            
            if is_full_path:
                final_path = video_input
            else:
                if folder_paths is not None:
                    try:
                        final_path = folder_paths.get_annotated_filepath(video_input)
                    except:
                        final_path = video_input
                else:
                    final_path = video_input
        
        if final_path and os.path.exists(final_path):
            try:
                m = hashlib.sha256()
                with open(final_path, 'rb') as f:
                    # Read first 1MB for quick hash (videos can be large)
                    m.update(f.read(1024 * 1024))
                return m.digest().hex()
            except:
                pass
        
        return float("NaN")
    
    @classmethod
    def VALIDATE_INPUTS(s, **kwargs):
        """Validate that video source is provided"""
        # Get the optional inputs
        video = kwargs.get('video', None)
        video_from_input = kwargs.get('video_from_input', "")
        video_path = kwargs.get('video_path', "")
        
        # Check if at least one video source is provided
        has_video_input = video is not None
        has_input_video = video_from_input and str(video_from_input).strip()
        has_custom_path = video_path and str(video_path).strip()
        
        # If video input is connected, validation passes
        if has_video_input:
            return True
        
        # If no video sources provided at all
        if not has_input_video and not has_custom_path:
            # This is okay - validation will fail at runtime with better error
            return True
        
        # Validate file paths if provided
        if has_custom_path:
            path_to_check = str(video_path).strip()
            if path_to_check and not os.path.exists(path_to_check):
                return f"Video file not found: {path_to_check}"
        
        elif has_input_video:
            video_input = str(video_from_input).strip()
            
            # Check if it's a full path or just a filename
            is_full_path = (
                ':' in video_input or
                video_input.startswith('\\\\') or
                video_input.startswith('/') or
                '/' in video_input or
                '\\' in video_input
            )
            
            if is_full_path:
                path_to_check = video_input
            else:
                # Try to get from input directory
                if folder_paths is not None:
                    try:
                        path_to_check = folder_paths.get_annotated_filepath(video_input)
                    except:
                        path_to_check = video_input
                else:
                    path_to_check = video_input
            
            # Check if file exists
            if path_to_check and not os.path.exists(path_to_check):
                return f"Video file not found: {path_to_check}"
        
        return True

