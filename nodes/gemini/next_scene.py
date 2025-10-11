# Gemini Next Scene Node

import os
import re
import random
import torch
import numpy as np
from PIL import Image
import google.generativeai as genai

from ...utils.constants import CUSTOM_CATEGORY, gemini_models


class GeminiNextScene:
    """
    Generates the next scene in a visual narrative using Google Gemini.
    Takes an original prompt and current frame image, then creates a cinematic transition
    to the next scene with camera movements, framing evolution, and atmospheric shifts.
    """
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=self.gemini_api_key)
        
        # Load the custom prompt template
        prompt_file = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "custom_prompts", "next_scene.txt")
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                self.system_prompt = f.read()
        except Exception as e:
            print(f"Warning: Could not load next_scene.txt: {e}")
            self.system_prompt = "Generate the next scene in this visual narrative."

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "original_prompt": ("STRING", {"multiline": True, "default": ""}),
                "gemini_model": (gemini_models, {"default": "gemini-2.5-flash"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "focus_on": (
                    ["Automatic", "Camera Movement", "Framing Evolution", "Environmental Reveals", "Atmospheric Shifts"],
                    {"default": "Automatic"}
                ),
                "transition_intensity": (
                    ["Subtle", "Moderate", "Dramatic"],
                    {"default": "Moderate"}
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("next_scene_prompt", "short_description")
    FUNCTION = "generate_next_scene"
    CATEGORY = CUSTOM_CATEGORY

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    def generate_next_scene(
        self,
        image,
        original_prompt="",
        gemini_model="gemini-2.5-flash",
        seed=-1,
        randomize_each_run=True,
        focus_on="Automatic",
        transition_intensity="Moderate"
    ):
        try:
            # Handle seed for randomization
            if randomize_each_run and seed == -1:
                # Generate a new random seed each time
                current_seed = random.randint(0, 0xffffffffffffffff)
            elif seed == -1:
                # Use a fixed seed for reproducibility
                current_seed = 12345
            else:
                # Use provided seed
                current_seed = seed
            
            # Set random seed for consistent randomization within this call
            random.seed(current_seed)
            
            print("\n" + "="*80)
            print("🎬 GEMINI NEXT SCENE - Processing")
            print("="*80)
            print(f"📝 Original Prompt: {original_prompt if original_prompt else '(Analyzing image only)'}")
            print(f"🤖 Model: {gemini_model}")
            print(f"🎯 Focus: {focus_on}")
            print(f"⚡ Intensity: {transition_intensity}")
            print(f"🎲 Using seed: {current_seed}")
            print("-"*80)
            
            # Convert tensor to PIL image
            if len(image.shape) == 4:
                pil_image = self.tensor2pil(image[0])
            else:
                pil_image = self.tensor2pil(image)

            print(f"🖼️  Image size: {pil_image.size}")
            print("🔄 Sending request to Gemini API...")

            # Prepare the prompt - handle both with and without original prompt
            if original_prompt and original_prompt.strip():
                # Use the full system prompt with original prompt context
                full_prompt = self.system_prompt.replace("##ORIGINAL_PROMPT##", original_prompt)
            else:
                # Create a simplified prompt that just analyzes the image
                full_prompt = """You are an expert cinematographer and visual storytelling assistant. 

Analyze the provided image and generate a detailed description of the "Next Scene" in a cinematic narrative.

First, understand what's happening in the current image, then create a natural, cinematic transition to the next scene.

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
1. Describes a specific camera movement or framing change
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

            # Configure safety settings to allow creative content
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

            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)

            response = model.generate_content([full_prompt, pil_image])
            
            # Check if response was blocked
            if not response.candidates:
                print("⚠️  WARNING: Gemini returned no candidates!")
                print("This usually means the content was completely blocked.")
                result = "Next Scene: The camera pulls back to reveal more of the surrounding environment, as lighting shifts to create a different mood and atmosphere."
                short_description = result
            elif response.candidates[0].finish_reason != 1:  # 1 = STOP (normal completion)
                print(f"⚠️  WARNING: Response did not complete normally. Finish reason: {response.candidates[0].finish_reason}")
                print("Checking safety ratings...")
                
                # Try to get safety ratings
                try:
                    if hasattr(response.candidates[0], 'safety_ratings'):
                        print("\n🛡️  SAFETY RATINGS:")
                        for rating in response.candidates[0].safety_ratings:
                            print(f"  - {rating.category}: {rating.probability}")
                except:
                    pass
                
                # Try to get any partial text
                try:
                    result = response.text.strip()
                    print(f"⚠️  Got partial response: {result[:100]}...")
                except:
                    # If completely blocked, provide a generic fallback
                    result = "Next Scene: The camera smoothly transitions to reveal a wider view of the scene, with dynamic lighting and atmospheric changes creating visual interest."
                    print("⚠️  Using fallback response due to content filtering.")
                
                short_description = result[:150]
            else:
                # Normal successful response
                result = response.text.strip()
                
                # Extract a shorter version (first sentence or first 150 chars)
                sentences = re.split(r'(?<=[.!?])\s+', result)
                short_description = sentences[0] if sentences else result[:150]

                print("✅ Response received!")
                print("-"*80)
                print("🎬 NEXT SCENE PROMPT (Full):")
                print("-"*80)
                print(result)
                print("-"*80)
                print("📋 Short Description:")
                print(short_description)
                print("="*80 + "\n")

            return (result, short_description)

        except Exception as e:
            print(f"\n❌ An error occurred in GeminiNextScene: {e}")
            
            # Try to provide more helpful error information
            if "safety_ratings" in str(e).lower() or "blocked" in str(e).lower():
                print("\n🛡️  CONTENT FILTER ISSUE:")
                print("The image or prompt was blocked by Gemini's safety filters.")
                print("\nPossible solutions:")
                print("1. Try a different image (avoid close-ups of people/faces)")
                print("2. Try adding a more neutral original_prompt")
                print("3. Use gemini-2.5-pro model which may be less restrictive")
                print("4. The image may contain content Gemini won't process")
            
            print("="*80 + "\n")
            error_message = f"Error generating next scene: {str(e)}"
            return (error_message, error_message)
