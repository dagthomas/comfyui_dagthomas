# Universal Vision Cloner Node - Model Agnostic

import os
import random
import json
import base64
import io
import numpy as np
import torch
from PIL import Image

# GPT imports (optional)
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

# Gemini imports (optional)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

# Anthropic imports (optional)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    anthropic = None

from ...utils.constants import CUSTOM_CATEGORY, gpt_models, gemini_models, grok_models, claude_models
from ...utils.image_utils import tensor2pil, pil2tensor


class UniversalVisionCloner:
    """
    Universal model-agnostic vision cloner that supports GPT, Gemini, and other vision models.
    Combines image analysis with prompt generation capabilities.
    """
    
    def __init__(self):
        # Don't initialize clients here - do it lazily when needed
        self.openai_client = None
        self.grok_client = None
        self.claude_client = None
        self.gemini_configured = False

    @classmethod
    def INPUT_TYPES(s):
        # Combine available vision models from all providers
        vision_models = ["auto-detect"]
        
        # Add GPT vision models
        vision_models += [f"gpt:{model}" for model in gpt_models if "vision" in model.lower() or "4o" in model.lower() or "4-turbo" in model.lower()]
        
        # Add Gemini models (all have vision)
        vision_models += [f"gemini:{model}" for model in gemini_models]
        
        # Add Grok vision models
        grok_vision_models = ["grok-2-vision-1212", "grok-4-0709", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning"]
        vision_models += [f"grok:{model}" for model in grok_models if model in grok_vision_models]
        
        # Add Claude models (all have vision)
        vision_models += [f"claude:{model}" for model in claude_models]
        
        # If no specific vision models found, include all models
        if len(vision_models) == 1:  # Only "auto-detect"
            vision_models = (["auto-detect"] + 
                           [f"gpt:{model}" for model in gpt_models] + 
                           [f"gemini:{model}" for model in gemini_models] +
                           [f"grok:{model}" for model in grok_models] +
                           [f"claude:{model}" for model in claude_models])
        
        return {
            "required": {
                "images": ("IMAGE",),
                "model": (vision_models, {"default": "auto-detect"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "analysis_mode": (
                    ["Detailed Analysis", "Style Cloning", "Scene Description", "Creative Interpretation", "Custom"], 
                    {"default": "Detailed Analysis"}
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "output_format": (
                    ["Text Only", "JSON Structure", "Formatted Prompt"],
                    {"default": "Formatted Prompt"}
                ),
                "detail_level": (
                    ["Brief", "Moderate", "Detailed", "Very Detailed"],
                    {"default": "Detailed"}
                ),
                "temperature": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE", "STRING")
    RETURN_NAMES = ("formatted_output", "raw_response", "faded_image", "model_used")
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def auto_detect_model(self):
        """Auto-detect the best available vision model"""
        # Check available API keys in order of preference (best vision models first)
        if ANTHROPIC_AVAILABLE and (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")):
            return "claude:claude-sonnet-4.5"
        elif OPENAI_AVAILABLE and (os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")):
            return "grok:grok-2-vision-1212"
        elif OPENAI_AVAILABLE and os.environ.get("OPENAI_API_KEY"):
            return "gpt:gpt-4o"
        elif GEMINI_AVAILABLE and os.environ.get("GEMINI_API_KEY"):
            return "gemini:gemini-2.5-flash"
        else:
            missing_deps = []
            if not OPENAI_AVAILABLE:
                missing_deps.append("openai library")
            if not GEMINI_AVAILABLE:
                missing_deps.append("google-generativeai library")
            if not ANTHROPIC_AVAILABLE:
                missing_deps.append("anthropic library")
            
            missing_keys = []
            if not os.environ.get("OPENAI_API_KEY"):
                missing_keys.append("OPENAI_API_KEY")
            if not os.environ.get("GEMINI_API_KEY"):
                missing_keys.append("GEMINI_API_KEY")
            
            error_msg = "No available models found. Missing: "
            if missing_deps:
                error_msg += f"Dependencies: {', '.join(missing_deps)}. "
            if missing_keys:
                error_msg += f"API keys: {', '.join(missing_keys)}."
            
            raise ValueError(error_msg)

    def get_analysis_prompt(self, analysis_mode, detail_level, output_format):
        """Generate analysis prompt based on mode and preferences"""
        
        detail_instructions = {
            "Brief": "Keep the analysis concise and focused on key elements (50-100 words).",
            "Moderate": "Provide a balanced analysis with good detail (100-200 words).",
            "Detailed": "Create a comprehensive analysis with rich visual details (200-300 words).",
            "Very Detailed": "Generate an extremely detailed analysis with extensive visual information (300+ words)."
        }
        
        mode_prompts = {
            "Detailed Analysis": """Analyze this image in detail, focusing on:
- Visual composition and framing
- Subject descriptions (people, objects, characters)
- Setting and environment details
- Lighting, shadows, and atmosphere
- Color palette and mood
- Style and artistic techniques
- Any text or symbolic elements
- Camera angle and perspective""",

            "Style Cloning": """Analyze this image to create a detailed style description that could be used to recreate similar images:
- Artistic style and technique
- Color grading and palette choices
- Lighting setup and quality
- Composition and framing approach
- Visual effects and post-processing
- Mood and atmosphere creation
- Technical camera settings (if apparent)""",

            "Scene Description": """Describe this scene as if creating a prompt for image generation:
- Main subjects and their positions
- Background and environment
- Actions and interactions happening
- Mood and emotional tone
- Visual style and aesthetic
- Important details that define the scene""",

            "Creative Interpretation": """Provide a creative interpretation of this image:
- Artistic vision and concept
- Emotional impact and storytelling
- Symbolic or metaphorical elements
- Creative techniques used
- Unique aspects that make it compelling
- How it might inspire new creative work""",

            "Custom": "Follow the custom prompt instructions precisely."
        }
        
        if output_format == "JSON Structure":
            json_structure = """{
"title": "A descriptive title for the image",
"subjects": ["List of main subjects/characters"],
"setting": "Description of the environment/location",
"style": "Artistic style and visual approach",
"colors": ["Dominant colors and palette"],
"lighting": "Lighting conditions and quality",
"mood": "Emotional tone and atmosphere",
"composition": "Framing and visual arrangement",
"details": ["Important specific details"],
"prompt_suggestion": "A refined prompt for recreating similar images"
}"""
            base_prompt = f"{mode_prompts.get(analysis_mode, mode_prompts['Detailed Analysis'])}\n\n{detail_instructions.get(detail_level, '')}\n\nProvide your analysis in this JSON format:\n{json_structure}"
        else:
            base_prompt = f"{mode_prompts.get(analysis_mode, mode_prompts['Detailed Analysis'])}\n\n{detail_instructions.get(detail_level, '')}"
        
        return base_prompt

    def fade_images(self, images, fade_percentage=15.0):
        """Fade multiple images together with blending"""
        if len(images) < 2:
            return images[0] if images else None

        # Determine orientation based on aspect ratio
        aspect_ratio = images[0].width / images[0].height
        vertical_stack = aspect_ratio > 1

        if vertical_stack:
            # Vertical stacking for wider images
            fade_height = int(images[0].height * (fade_percentage / 100))
            total_height = sum(img.height for img in images) - fade_height * (len(images) - 1)
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
                                pixel1 = images[i - 1].getpixel((x, images[i - 1].height - fade_height + y))
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel((x, y_offset + y), blended_pixel)

                    combined_image.paste(
                        img.crop((0, fade_height, img.width, img.height)),
                        (0, y_offset + fade_height),
                    )
                    y_offset += img.height - fade_height
        else:
            # Horizontal stacking for taller images
            fade_width = int(images[0].width * (fade_percentage / 100))
            total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
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
                                pixel1 = images[i - 1].getpixel((images[i - 1].width - fade_width + x, y))
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel((x_offset + x, y), blended_pixel)

                    combined_image.paste(
                        img.crop((fade_width, 0, img.width, img.height)),
                        (x_offset + fade_width, 0),
                    )
                    x_offset += img.width - fade_width

        return combined_image

    def encode_image(self, image):
        """Encode image to base64 for GPT"""
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def format_gpt_response(self, content, output_format):
        """Format GPT response based on output format"""
        if output_format == "JSON Structure":
            # Try to parse and reformat JSON
            try:
                if content.startswith("```json") and content.endswith("```"):
                    json_str = content[7:-3].strip()
                else:
                    json_str = content
                
                data = json.loads(json_str)
                
                if output_format == "Formatted Prompt" and "prompt_suggestion" in data:
                    return data["prompt_suggestion"]
                else:
                    return json.dumps(data, indent=2)
            except:
                return content
        
        return content

    def extract_formatted_prompt(self, data):
        """Extract a formatted prompt from structured data"""
        if isinstance(data, dict):
            elements = []
            
            if "title" in data:
                elements.append(data["title"])
            
            if "subjects" in data and isinstance(data["subjects"], list):
                elements.extend(data["subjects"])
            
            if "setting" in data:
                elements.append(data["setting"])
            
            if "style" in data:
                elements.append(data["style"])
            
            if "colors" in data and isinstance(data["colors"], list):
                elements.append(f"palette ({', '.join(data['colors'])})")
            
            if "lighting" in data:
                elements.append(data["lighting"])
            
            if "mood" in data:
                elements.append(data["mood"])
            
            if "details" in data and isinstance(data["details"], list):
                elements.extend(data["details"])
            
            return ", ".join(item for item in elements if item)
        
        return str(data)

    def analyze_with_gpt(self, model_name, image, prompt, temperature, seed):
        """Analyze image using OpenAI GPT"""
        try:
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not available. Install with: pip install openai")
            
            # Lazy initialization of OpenAI client
            if self.openai_client is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.openai_client = OpenAI(api_key=api_key)
            
            # Extract model name (remove "gpt:" prefix)
            gpt_model = model_name.replace("gpt:", "")
            
            # Encode image
            base64_image = self.encode_image(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        }
                    ]
                }
            ]
            
            response = self.openai_client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                temperature=temperature,
                top_p=0.95,
                seed=seed if seed != -1 else None,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"GPT vision analysis failed: {str(e)}")

    def analyze_with_gemini(self, model_name, image, prompt, temperature):
        """Analyze image using Google Gemini"""
        try:
            if not GEMINI_AVAILABLE:
                raise ValueError("Google Generative AI library not available. Install with: pip install google-generativeai")
            
            # Lazy initialization of Gemini
            if not self.gemini_configured:
                api_key = os.environ.get("GEMINI_API_KEY")
                if not api_key:
                    raise ValueError("GEMINI_API_KEY environment variable not set")
                genai.configure(api_key=api_key)
                self.gemini_configured = True
            
            # Extract model name (remove "gemini:" prefix)
            gemini_model = model_name.replace("gemini:", "")
            
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]

            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)
            
            print(f"🔄 Sending to Gemini vision model: {gemini_model}")
            response = model.generate_content([prompt, image])
            
            print("📥 Gemini vision response received!")
            result = response.text
            print(f"📝 Response length: {len(result)} characters")
            
            return result.strip()
            
        except Exception as e:
            print(f"❌ Gemini vision error: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            return error_message

    def analyze_with_grok(self, model_name, image, prompt, temperature, seed):
        """Analyze image using xAI Grok"""
        try:
            if not OPENAI_AVAILABLE:
                raise ValueError("OpenAI library not available. Install with: pip install openai")
            
            # Lazy initialization of Grok client
            if self.grok_client is None:
                api_key = os.environ.get("XAI_API_KEY") or os.environ.get("GROK_API_KEY")
                if not api_key:
                    raise ValueError("XAI_API_KEY or GROK_API_KEY environment variable not set")
                self.grok_client = OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1"
                )
            
            # Extract model name (remove "grok:" prefix)
            grok_model = model_name.replace("grok:", "")
            
            # Encode image
            base64_image = self.encode_image(image)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                        }
                    ]
                }
            ]
            
            print(f"🔄 Sending to Grok vision model: {grok_model}")
            response = self.grok_client.chat.completions.create(
                model=grok_model,
                messages=messages,
                max_tokens=2000,
                temperature=temperature,
                seed=seed if seed != -1 else None,
            )
            
            result = response.choices[0].message.content
            print(f"📥 Grok response: {len(result)} characters")
            return result.strip()
            
        except Exception as e:
            print(f"❌ Grok vision error: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            return error_message

    def analyze_with_claude(self, model_name, image, prompt, temperature):
        """Analyze image using Anthropic Claude"""
        try:
            if not ANTHROPIC_AVAILABLE:
                raise ValueError("Anthropic library not available. Install with: pip install anthropic")
            
            # Lazy initialization of Claude client
            if self.claude_client is None:
                api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("CLAUDE_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY or CLAUDE_API_KEY environment variable not set")
                self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Extract model name (remove "claude:" prefix)
            claude_model = model_name.replace("claude:", "")
            
            # Encode image
            base64_image = self.encode_image(image)
            
            print(f"🔄 Sending to Claude vision model: {claude_model}")
            response = self.claude_client.messages.create(
                model=claude_model,
                max_tokens=2000,
                temperature=temperature,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": base64_image
                                }
                            }
                        ]
                    }
                ]
            )
            
            result = response.content[0].text
            print(f"📥 Claude response: {len(result)} characters")
            return result.strip()
            
        except Exception as e:
            print(f"❌ Claude vision error: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            return error_message

    def analyze_images(
        self,
        images,
        model="auto-detect",
        fade_percentage=15.0,
        analysis_mode="Detailed Analysis",
        seed=-1,
        randomize_each_run=True,
        custom_prompt="",
        output_format="Formatted Prompt",
        detail_level="Detailed",
        temperature=-1.0,
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
            
            # Auto-detect model if needed
            if model == "auto-detect":
                model = self.auto_detect_model()
            
            print(f"🤖 Universal Vision Cloner using model: {model}")
            print(f"🎲 Using seed: {current_seed}")
            print(f"🎯 Analysis mode: {analysis_mode}")
            
            # Set temperature based on mode if not manually set
            if temperature == -1.0:
                temp_map = {
                    "Creative Interpretation": 1.1,
                    "Style Cloning": 0.9,
                    "Detailed Analysis": 0.7,
                    "Scene Description": 0.8,
                    "Custom": 0.8
                }
                temperature = temp_map.get(analysis_mode, 0.8)
            
            # Handle single image or multiple images
            if len(images.shape) == 3:  # Single image
                pil_images = [tensor2pil(images)]
            else:  # Multiple images
                pil_images = [tensor2pil(img) for img in images]

            # Combine images with fading
            combined_image = self.fade_images(pil_images, fade_percentage)
            
            # Build the prompt
            if custom_prompt.strip():
                final_prompt = custom_prompt
            else:
                final_prompt = self.get_analysis_prompt(analysis_mode, detail_level, output_format)
            
            # Add variation instruction for randomization
            if randomize_each_run:
                final_prompt += f"\n\nProvide unique creative variations each time while maintaining accuracy. Use seed {current_seed} for consistent but varied analysis."
            
            print(f"📤 Analyzing image with {model}")
            
            # Generate based on model type
            if model.startswith("gpt:"):
                raw_result = self.analyze_with_gpt(model, combined_image, final_prompt, temperature, current_seed)
                formatted_result = self.format_gpt_response(raw_result, output_format)
            elif model.startswith("gemini:"):
                raw_result = self.analyze_with_gemini(model, combined_image, final_prompt, temperature)
                formatted_result = raw_result  # Gemini typically returns clean text
            elif model.startswith("grok:"):
                raw_result = self.analyze_with_grok(model, combined_image, final_prompt, temperature, current_seed)
                formatted_result = self.format_gpt_response(raw_result, output_format)  # Grok uses OpenAI format
            elif model.startswith("claude:"):
                raw_result = self.analyze_with_claude(model, combined_image, final_prompt, temperature)
                formatted_result = raw_result  # Claude typically returns clean text
            else:
                raise ValueError(f"Unsupported model format: {model}")
            
            # Create final formatted output
            if output_format == "Formatted Prompt":
                try:
                    # Try to parse as JSON first for better formatting
                    if raw_result.strip().startswith('{'):
                        data = json.loads(raw_result)
                        final_output = self.extract_formatted_prompt(data)
                    else:
                        final_output = formatted_result
                except:
                    final_output = formatted_result
            else:
                final_output = formatted_result
            
            # Convert combined image back to tensor
            faded_image_tensor = pil2tensor(combined_image)
            
            print(f"✅ Generated {len(final_output)} characters of analysis")
            
            return (final_output, raw_result, faded_image_tensor, model)

        except Exception as e:
            print(f"❌ An error occurred in Universal Vision Cloner: {e}")
            error_message = f"Error analyzing image: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            error_tensor = pil2tensor(error_image)
            return (error_message, error_message, error_tensor, model if 'model' in locals() else "unknown")
