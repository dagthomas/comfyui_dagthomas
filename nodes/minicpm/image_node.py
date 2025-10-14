# MiniCPM-V Image Node
# Implements MiniCPM-V-4.5 for image understanding

import torch
import numpy as np
from PIL import Image
from ...utils.constants import CUSTOM_CATEGORY

# Lazy imports for heavy dependencies
transformers = None

def lazy_import_dependencies():
    """Lazy import of heavy dependencies only when needed"""
    global transformers
    if transformers is None:
        try:
            import transformers as tf
            transformers = tf
        except ImportError:
            raise ImportError("transformers library not found. Please install with: pip install transformers")
    
    return transformers


class MiniCPMImageNode:
    """
    MiniCPM-V 4.5 Image Understanding Node
    
    Uses the MiniCPM-V-4.5 model from OpenBMB for single or multiple image understanding.
    Supports fast thinking for efficient frequent usage and deep thinking for complex tasks.
    """
    
    # Class-level model cache to avoid reloading
    _model_cache = {}
    _tokenizer_cache = {}
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "question": ("STRING", {
                    "multiline": True,
                    "default": "Describe this image in detail."
                }),
                "model_name": (
                    ["openbmb/MiniCPM-V-4_5", "openbmb/MiniCPM-o-2_6"],
                    {"default": "openbmb/MiniCPM-V-4_5"}
                ),
                "precision": ([
                    "bfloat16",
                    "float16",
                ], {
                    "default": "bfloat16",
                    "tooltip": "float16 uses slightly less memory. bfloat16 is more stable."
                }),
                "enable_thinking": ("BOOLEAN", {
                    "default": False
                }),
                "stream": ("BOOLEAN", {
                    "default": False
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
                "conversation_history": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("response", "conversation_history")
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY
    
    @staticmethod
    def tensor2pil(image):
        """Convert ComfyUI tensor to PIL Image"""
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )
    
    def load_model(self, model_name, device, precision="bfloat16"):
        """Load or retrieve cached model and tokenizer"""
        tf = lazy_import_dependencies()
        
        cache_key = f"{model_name}_{device}_{precision}"
        
        # Check if model is already loaded
        if cache_key in self._model_cache:
            print(f"✅ Using cached model: {model_name} ({precision})")
            return self._model_cache[cache_key], self._tokenizer_cache[cache_key]
        
        print("\n" + "="*80)
        print(f"📥 LOADING MODEL: {model_name}")
        print(f"⚙️  Precision: {precision}")
        print("="*80)
        
        try:
            # Determine torch dtype
            if precision == "float16":
                torch_dtype = torch.float16
                print("💾 Using FP16 precision")
            else:
                torch_dtype = torch.bfloat16
                print("💾 Using BF16 precision")
            
            # Load model with standard optimizations
            print("🔧 Loading model with sdpa attention...")
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
        """Unload model from memory"""
        cache_key = f"{model_name}_{device}_{precision}"
        
        if cache_key in self._model_cache:
            print(f"🗑️  Unloading model: {model_name}")
            del self._model_cache[cache_key]
            del self._tokenizer_cache[cache_key]
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            print("✅ Model unloaded from memory")
    
    def analyze_images(
        self,
        images,
        question="Describe this image in detail.",
        model_name="openbmb/MiniCPM-V-4_5",
        precision="bfloat16",
        enable_thinking=False,
        stream=False,
        device="cuda",
        unload_after_inference=False,
        conversation_history="",
    ):
        """
        Analyze image(s) using MiniCPM-V model
        
        Args:
            images: ComfyUI IMAGE tensor (can be multiple images)
            question: Question to ask about the image(s)
            model_name: Model to use
            enable_thinking: Enable thinking mode for complex reasoning
            stream: Stream the response (returns full text at end)
            device: Device to use (cuda/cpu)
            unload_after_inference: Whether to unload model after inference
            conversation_history: Previous conversation context (JSON format)
        
        Returns:
            response: Model's response
            conversation_history: Updated conversation history
        """
        try:
            # Load model and tokenizer
            model, tokenizer = self.load_model(model_name, device, precision)
            
            # Convert tensor images to PIL
            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]
            
            print("\n" + "="*80)
            print("🖼️  IMAGE ANALYSIS")
            print("="*80)
            print(f"📊 Number of images: {len(pil_images)}")
            print(f"📐 Image sizes: {[img.size for img in pil_images]}")
            print(f"❓ Question: {question}")
            print(f"🧠 Thinking mode: {'Enabled' if enable_thinking else 'Disabled'}")
            print("-"*80)
            
            # Parse conversation history if provided
            import json
            msgs = []
            if conversation_history:
                try:
                    msgs = json.loads(conversation_history)
                    print(f"💬 Loaded {len(msgs)} previous messages")
                except:
                    print("⚠️  Could not parse conversation history, starting fresh")
                    msgs = []
            
            # Add current question with images
            msgs.append({'role': 'user', 'content': pil_images + [question]})
            
            # Run inference
            if stream:
                print("🔄 Streaming response...")
                generated_text = ""
                answer_stream = model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                    stream=True
                )
                for new_text in answer_stream:
                    generated_text += new_text
                answer = generated_text
            else:
                print("⚡ Running inference...")
                answer = model.chat(
                    msgs=msgs,
                    tokenizer=tokenizer,
                    enable_thinking=enable_thinking,
                    stream=False
                )
            
            print("✅ INFERENCE COMPLETE")
            print("="*80)
            print("📝 RESPONSE:")
            print("-"*80)
            print(answer)
            print("="*80 + "\n")
            
            # Update conversation history
            msgs.append({"role": "assistant", "content": [answer]})
            updated_history = json.dumps(msgs, ensure_ascii=False)
            
            # Unload model if requested
            if unload_after_inference:
                self.unload_model(model_name, device, precision)
            
            return (answer, updated_history)
            
        except ImportError as e:
            error_msg = f"Missing dependency: {str(e)}\n\nPlease install required packages:\npip install transformers torch"
            print(f"❌ {error_msg}")
            return (error_msg, conversation_history)
            
        except Exception as e:
            error_msg = f"Error analyzing image: {str(e)}"
            print(f"❌ {error_msg}")
            import traceback
            traceback.print_exc()
            return (error_msg, conversation_history)

