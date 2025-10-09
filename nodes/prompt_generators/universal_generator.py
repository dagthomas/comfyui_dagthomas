# APNext Universal Generator Node - Model Agnostic

import os
import random
import google.generativeai as genai
from openai import OpenAI

from ...utils.constants import CUSTOM_CATEGORY, gpt_models, gemini_models


class APNextGenerator:
    """
    Universal model-agnostic prompt generator that supports GPT, Gemini, and other models.
    Includes seed functionality for reproducible variations.
    """
    
    def __init__(self):
        # Don't initialize clients here - do it lazily when needed
        self.openai_client = None
        self.gemini_configured = False

    @classmethod
    def INPUT_TYPES(s):
        # Combine available models from all providers
        all_models = ["auto-detect"] + [f"gpt:{model}" for model in gpt_models] + [f"gemini:{model}" for model in gemini_models]
        
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "model": (all_models, {"default": "auto-detect"}),
                "generation_mode": (
                    ["Creative", "Balanced", "Focused", "Custom"], 
                    {"default": "Balanced"}
                ),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "variation_instruction": (
                    "STRING", 
                    {"multiline": True, "default": "Generate different creative variations each time while maintaining the core concept and style."}
                ),
                "style_preference": (
                    ["Any", "Cinematic", "Photorealistic", "Artistic", "Abstract", "Vintage", "Modern"],
                    {"default": "Any"}
                ),
                "detail_level": (
                    ["Brief", "Moderate", "Detailed", "Very Detailed"],
                    {"default": "Detailed"}
                ),
                "temperature": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.1}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("generated_prompt", "model_used", "seed_used")
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def auto_detect_model(self):
        """Auto-detect the best available model"""
        # Check if Gemini API key is available first (often more reliable)
        if os.environ.get("GEMINI_API_KEY"):
            return "gemini:gemini-2.5-flash"
        # Check if OpenAI API key is available
        elif os.environ.get("OPENAI_API_KEY"):
            return "gpt:gpt-4o-mini"
        else:
            raise ValueError("No API keys found. Please set OPENAI_API_KEY or GEMINI_API_KEY")

    def get_base_prompt(self, generation_mode, detail_level, style_preference):
        """Generate base prompt based on preferences"""
        
        style_guidance = {
            "Cinematic": "with professional cinematography, dramatic lighting, and film-like composition",
            "Photorealistic": "with photorealistic detail, natural lighting, and realistic textures",
            "Artistic": "with artistic flair, creative composition, and stylized elements",
            "Abstract": "with abstract concepts, experimental composition, and conceptual elements",
            "Vintage": "with vintage aesthetics, retro styling, and nostalgic atmosphere",
            "Modern": "with contemporary styling, clean composition, and modern aesthetics",
            "Any": ""
        }
        
        detail_instructions = {
            "Brief": "Keep the description concise and focused on key elements (50-100 words).",
            "Moderate": "Provide a balanced description with good detail (100-200 words).",
            "Detailed": "Create a comprehensive description with rich visual details (200-300 words).",
            "Very Detailed": "Generate an extremely detailed description with extensive visual information (300+ words)."
        }
        
        mode_instructions = {
            "Creative": "Be highly creative and imaginative. Take creative liberties and add interesting, unexpected elements.",
            "Balanced": "Balance creativity with accuracy. Enhance the concept while staying true to the core idea.",
            "Focused": "Stay focused on the core concept. Enhance with relevant details without major creative additions.",
            "Custom": "Follow the custom prompt instructions precisely."
        }
        
        style_text = style_guidance.get(style_preference, "")
        detail_text = detail_instructions.get(detail_level, "")
        mode_text = mode_instructions.get(generation_mode, "")
        
        # Use a much simpler prompt to avoid triggering Gemini's safety filters
        base_prompt = f"""Create a detailed visual description for an AI image generation system. {mode_text} {detail_text}

Focus on:
- Visual composition and framing
- Character descriptions (if applicable)
- Setting and environment details
- Lighting and atmosphere
- Color palette and mood
- Style and artistic direction
{style_text}

Generate a prompt that would create compelling, high-quality images. Be specific about visual elements while maintaining artistic flow."""

        return base_prompt

    def generate_with_gpt(self, model_name, prompt, temperature, seed):
        """Generate using OpenAI GPT"""
        try:
            # Lazy initialization of OpenAI client
            if self.openai_client is None:
                api_key = os.environ.get("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self.openai_client = OpenAI(api_key=api_key)
            
            # Extract model name (remove "gpt:" prefix)
            gpt_model = model_name.replace("gpt:", "")
            
            response = self.openai_client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                temperature=temperature,
                top_p=0.95,
                seed=seed if seed != -1 else None,
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"GPT generation failed: {str(e)}")

    def generate_with_gemini(self, model_name, prompt, temperature):
        """Generate using Google Gemini"""
        try:
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
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            # Use EXACTLY the same pattern as your working nodes - no generation_config
            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)

            print(f"🔄 Sending to Gemini model: {gemini_model}")
            response = model.generate_content(prompt)
            
            print("📥 Gemini response received!")
            
            # EXACT same pattern as your working GeminiTextOnly node
            result = response.text
            print(f"📝 Response length: {len(result)} characters")
            return result.strip()
            
        except Exception as e:
            print(f"❌ Gemini error: {e}")
            print(f"📤 The prompt that caused the error was: {prompt}")
            # Simple fallback exactly like your working nodes
            error_message = f"Error occurred while processing the request: {str(e)}"
            return error_message

    def generate(
        self,
        input_text,
        model="auto-detect",
        generation_mode="Balanced",
        seed=-1,
        randomize_each_run=True,
        custom_prompt="",
        variation_instruction="Generate different creative variations each time while maintaining the core concept and style.",
        style_preference="Any",
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
            
            print(f"🤖 APNext Generator using model: {model}")
            print(f"🎲 Using seed: {current_seed}")
            print(f"🎯 Mode: {generation_mode}")
            
            # Set temperature based on mode if not manually set
            if temperature == -1.0:
                temp_map = {
                    "Creative": 1.1,
                    "Balanced": 0.9,
                    "Focused": 0.7,
                    "Custom": 0.8
                }
                temperature = temp_map.get(generation_mode, 0.9)
            
            # Build the prompt
            if custom_prompt.strip():
                base_prompt = custom_prompt
            else:
                base_prompt = self.get_base_prompt(generation_mode, detail_level, style_preference)
            
            # Create final prompt - use EXACT same approach as working GeminiTextOnly
            if model.startswith("gemini:"):
                # Use the exact same pattern as GeminiTextOnly node
                if custom_prompt.strip():
                    final_prompt = custom_prompt
                else:
                    final_prompt = f"Create a visual description: {input_text}"
            else:
                # More detailed prompt for GPT (which is less restrictive)
                if randomize_each_run and variation_instruction.strip():
                    base_prompt += f"\n\nVARIATION INSTRUCTION: {variation_instruction} Use seed {current_seed} for unique creative variations."
                
                final_prompt = f"{base_prompt}\n\nInput Description: {input_text}\n\nGenerate a detailed visual prompt:"
            
            # # LOG THE PROMPT BEING SENT
            # print("\n" + "="*60)
            # print(f"📤 PROMPT BEING SENT TO {model.upper()}:")
            # print("="*60)
            # print(final_prompt)
            # print("="*60)
            
            # Generate based on model type
            if model.startswith("gpt:"):
                result = self.generate_with_gpt(model, final_prompt, temperature, current_seed)
            elif model.startswith("gemini:"):
                result = self.generate_with_gemini(model, final_prompt, temperature)
            else:
                raise ValueError(f"Unsupported model format: {model}")
            
            print(f"✅ Generated {len(result)} characters")
            
            return (result, model, str(current_seed))

        except Exception as e:
            print(f"❌ An error occurred in APNext Generator: {e}")
            error_message = f"Error generating content: {str(e)}"
            return (error_message, model if 'model' in locals() else "unknown", str(current_seed) if 'current_seed' in locals() else "-1")
