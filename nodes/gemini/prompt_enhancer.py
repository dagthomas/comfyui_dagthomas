# Gemini Prompt Enhancer Node

import random
import os
import google.generativeai as genai

from ...utils.constants import CUSTOM_CATEGORY, CINEMATIC_TERMS, gemini_models


class GeminiPromptEnhancer:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if self.gemini_api_key:
            genai.configure(api_key=self.gemini_api_key)

    @classmethod
    def INPUT_TYPES(s):
        enhancement_modes = [
            "Random Mix (4-6 elements)",
            "Cinematic Focus",
            "Lighting Focus", 
            "Camera Focus",
            "Motion Focus",
            "Style Focus",
            "Full Enhancement",
            "LLM Only (No Random)"
        ]
        
        # Create dropdown options for each category
        visual_style_options = ["none", "random"] + CINEMATIC_TERMS["visual_style"]
        lighting_type_options = ["none", "random"] + CINEMATIC_TERMS["lighting_type"]
        light_source_options = ["none", "random"] + CINEMATIC_TERMS["light_source"]
        camera_angle_options = ["none", "random"] + CINEMATIC_TERMS["camera_angle"]
        shot_size_options = ["none", "random"] + CINEMATIC_TERMS["shot_size"]
        lens_type_options = ["none", "random"] + CINEMATIC_TERMS["lens_type"]
        color_tone_options = ["none", "random"] + CINEMATIC_TERMS["color_tone"]
        camera_movement_options = ["none", "random"] + CINEMATIC_TERMS["camera_movement"]
        time_of_day_options = ["none", "random"] + CINEMATIC_TERMS["time_of_day"]
        visual_effects_options = ["none", "random"] + CINEMATIC_TERMS["visual_effects"]
        composition_options = ["none", "random"] + CINEMATIC_TERMS["composition"]
        motion_options = ["none", "random"] + CINEMATIC_TERMS["motion"]
        character_emotion_options = ["none", "random"] + CINEMATIC_TERMS["character_emotion"]
        
        return {
            "required": {
                "base_prompt": ("STRING", {"multiline": True, "default": "A knight fighting a dragon"}),
                "enhancement_mode": (enhancement_modes, {"default": "Random Mix (4-6 elements)"}),
                "use_llm": ("BOOLEAN", {"default": True}),
                "gemini_model": (gemini_models, {"default": "gemini-flash-latest"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 999999}),
            },
            "optional": {
                "custom_llm_prompt": ("STRING", {"multiline": True, "default": ""}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "visual_style": (visual_style_options, {"default": "random"}),
                "lighting_type": (lighting_type_options, {"default": "random"}),
                "light_source": (light_source_options, {"default": "random"}),
                "camera_angle": (camera_angle_options, {"default": "random"}),
                "shot_size": (shot_size_options, {"default": "random"}),
                "lens_type": (lens_type_options, {"default": "random"}),
                "color_tone": (color_tone_options, {"default": "random"}),
                "camera_movement": (camera_movement_options, {"default": "random"}),
                "time_of_day": (time_of_day_options, {"default": "random"}),
                "visual_effects": (visual_effects_options, {"default": "random"}),
                "composition": (composition_options, {"default": "random"}),
                "motion": (motion_options, {"default": "random"}),
                "character_emotion": (character_emotion_options, {"default": "random"}),
                # APNext chain input
                "apnext_chain": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "random_enhanced", "llm_enhanced")
    FUNCTION = "enhance_prompt"
    CATEGORY = CUSTOM_CATEGORY

    def enhance_prompt_basic(self, base_prompt: str, mode: str, intensity: float = 1.0, seed: int = 0) -> str:
        """
        Takes a simple user prompt and enhances it with randomly selected cinematic terms.
        """
        random.seed(seed)
        
        if mode == "Random Mix (4-6 elements)":
            num_enhancements = max(1, int(random.randint(4, 6) * intensity))
            all_categories = list(CINEMATIC_TERMS.keys())
            chosen_categories = random.sample(all_categories, min(num_enhancements, len(all_categories)))
        elif mode == "Cinematic Focus":
            chosen_categories = ["shot_size", "camera_angle", "lighting_type", "composition"]
        elif mode == "Lighting Focus":
            chosen_categories = ["lighting_type", "light_source", "time_of_day", "color_tone"]
        elif mode == "Camera Focus":
            chosen_categories = ["camera_angle", "shot_size", "lens_type", "camera_movement"]
        elif mode == "Motion Focus":
            chosen_categories = ["motion", "camera_movement", "character_emotion", "visual_effects"]
        elif mode == "Style Focus":
            chosen_categories = ["visual_style", "color_tone", "visual_effects", "composition"]
        elif mode == "Full Enhancement":
            chosen_categories = list(CINEMATIC_TERMS.keys())
            num_enhancements = max(1, int(len(chosen_categories) * intensity))
            chosen_categories = random.sample(chosen_categories, min(num_enhancements, len(chosen_categories)))
        else:  # LLM Only
            return base_prompt
            
        # Build the list of enhancement phrases
        enhancements = []
        for category in chosen_categories:
            term = random.choice(CINEMATIC_TERMS[category])
            enhancements.append(term)
            
        # Combine the base prompt with the new enhancements
        enhancements_str = ", ".join(enhancements)
        
        # Create the final, complex prompt
        complex_prompt = f"{base_prompt}, {enhancements_str}."
        
        return complex_prompt

    def enhance_prompt_with_selections(self, base_prompt: str, mode: str, intensity: float, seed: int, user_selections: dict) -> str:
        """
        Enhanced version that respects user dropdown selections for each category.
        """
        random.seed(seed)
        
        # Build enhancements list based on user selections and mode
        enhancements = []
        
        # If mode is LLM Only, return base prompt
        if mode == "LLM Only (No Random)":
            return base_prompt
            
        # Determine which categories to use based on mode
        if mode == "Random Mix (4-6 elements)":
            num_enhancements = max(1, int(random.randint(4, 6) * intensity))
            all_categories = list(CINEMATIC_TERMS.keys())
            chosen_categories = random.sample(all_categories, min(num_enhancements, len(all_categories)))
        elif mode == "Cinematic Focus":
            chosen_categories = ["shot_size", "camera_angle", "lighting_type", "composition"]
        elif mode == "Lighting Focus":
            chosen_categories = ["lighting_type", "light_source", "time_of_day", "color_tone"]
        elif mode == "Camera Focus":
            chosen_categories = ["camera_angle", "shot_size", "lens_type", "camera_movement"]
        elif mode == "Motion Focus":
            chosen_categories = ["motion", "camera_movement", "character_emotion", "visual_effects"]
        elif mode == "Style Focus":
            chosen_categories = ["visual_style", "color_tone", "visual_effects", "composition"]
        elif mode == "Full Enhancement":
            chosen_categories = list(CINEMATIC_TERMS.keys())
            num_enhancements = max(1, int(len(chosen_categories) * intensity))
            chosen_categories = random.sample(chosen_categories, min(num_enhancements, len(chosen_categories)))
        else:
            chosen_categories = []
            
        # Process each chosen category
        for category in chosen_categories:
            user_choice = user_selections.get(category, "random")
            
            if user_choice == "none":
                # Skip this category
                continue
            elif user_choice == "random":
                # Pick randomly from this category
                if category in CINEMATIC_TERMS:
                    term = random.choice(CINEMATIC_TERMS[category])
                    enhancements.append(term)
            else:
                # Use the specific user selection
                if user_choice in CINEMATIC_TERMS.get(category, []):
                    enhancements.append(user_choice)
                    
        # Combine the base prompt with the selected enhancements
        if enhancements:
            enhancements_str = ", ".join(enhancements)
            complex_prompt = f"{base_prompt}, {enhancements_str}."
        else:
            complex_prompt = base_prompt
            
        return complex_prompt

    def enhance_with_llm(self, prompt: str, custom_prompt: str = "", gemini_model: str = "gemini-flash-latest") -> str:
        """
        Enhance the prompt using Gemini LLM for more sophisticated enhancement.
        """
        try:
            if not self.gemini_api_key:
                print("ERROR: GEMINI_API_KEY not set, returning original prompt")
                return prompt
            
            if custom_prompt:
                system_prompt = custom_prompt
            else:
                system_prompt = """Enhance this prompt for AI video generation models like Runway, Luma, or Kling. Add sophisticated cinematic language, advanced camera techniques (dolly shots, crane movements, rack focus), dynamic lighting details (volumetric rays, chiaroscuro, rim lighting), and vivid atmospheric descriptions. Improve narrative flow and emotional depth. Use professional cinematography terms and color grading language. Incorporate motion dynamics, texture details, and environmental storytelling. Make descriptions more immersive and visually striking. Always enhance further, even if already detailed. CRITICAL: Keep your enhanced prompt between 80-120 words maximum. Output only the enhanced prompt, no explanations."""

            full_prompt = f"{system_prompt}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"

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
            response = model.generate_content(full_prompt)
            
            enhanced_result = response.text.strip()
            
            return enhanced_result
            
        except Exception as e:
            print(f"ERROR: LLM enhancement failed: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            return prompt  # Return original if LLM fails

    def enhance_with_llm_and_apnext(self, prompt: str, custom_prompt: str, gemini_model: str, apnext_context: str) -> str:
        """
        Enhanced LLM method that incorporates APNext chain input for more contextual enhancement.
        """
        try:
            if not self.gemini_api_key:
                print("ERROR: GEMINI_API_KEY not set, returning original prompt")
                return prompt
            
            if custom_prompt:
                system_prompt = custom_prompt
            else:
                base_prompt = """Enhance this prompt for AI video generation models like Runway, Luma, or Kling. Add sophisticated cinematic language, advanced camera techniques (dolly shots, crane movements, rack focus), dynamic lighting details (volumetric rays, chiaroscuro, rim lighting), and vivid atmospheric descriptions. Improve narrative flow and emotional depth. Use professional cinematography terms and color grading language. Incorporate motion dynamics, texture details, and environmental storytelling. Make descriptions more immersive and visually striking. Always enhance further, even if already detailed. CRITICAL: Keep your enhanced prompt between 80-120 words maximum. Output only the enhanced prompt, no explanations."""
                
                if apnext_context:
                    system_prompt = f"""{base_prompt}

ADDITIONAL CONTEXT from APNext chain:
{apnext_context}

Incorporate these elements naturally into your enhancement where relevant."""
                else:
                    system_prompt = base_prompt

            full_prompt = f"{system_prompt}\n\nOriginal prompt: {prompt}\n\nEnhanced prompt:"

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
            response = model.generate_content(full_prompt)
            
            enhanced_result = response.text.strip()
            
            return enhanced_result
            
        except Exception as e:
            print(f"ERROR: LLM+APNext enhancement failed: {e}")
            print(f"ERROR: Exception type: {type(e).__name__}")
            import traceback
            print(f"ERROR: Full traceback: {traceback.format_exc()}")
            return prompt  # Return original if LLM fails

    def enhance_prompt(
        self,
        base_prompt: str,
        enhancement_mode: str = "Random Mix (4-6 elements)",
        use_llm: bool = True,
        gemini_model: str = "gemini-flash-latest",
        seed: int = 0,
        custom_llm_prompt: str = "",
        intensity: float = 1.0,
        visual_style: str = "random",
        lighting_type: str = "random",
        light_source: str = "random",
        camera_angle: str = "random",
        shot_size: str = "random",
        lens_type: str = "random",
        color_tone: str = "random",
        camera_movement: str = "random",
        time_of_day: str = "random",
        visual_effects: str = "random",
        composition: str = "random",
        motion: str = "random",
        character_emotion: str = "random",
        apnext_chain: str = "",
    ) -> tuple:
        """
        Main function that combines random enhancement with optional LLM enhancement.
        """
        
        # Collect user selections
        user_selections = {
            "visual_style": visual_style,
            "lighting_type": lighting_type,
            "light_source": light_source,
            "camera_angle": camera_angle,
            "shot_size": shot_size,
            "lens_type": lens_type,
            "color_tone": color_tone,
            "camera_movement": camera_movement,
            "time_of_day": time_of_day,
            "visual_effects": visual_effects,
            "composition": composition,
            "motion": motion,
            "character_emotion": character_emotion,
        }
        
        # Generate random enhanced version
        random_enhanced = self.enhance_prompt_with_selections(
            base_prompt, enhancement_mode, intensity, seed, user_selections
        )
        
        # Generate LLM enhanced version if requested
        if use_llm:
            if apnext_chain:
                llm_enhanced = self.enhance_with_llm_and_apnext(
                    random_enhanced, custom_llm_prompt, gemini_model, apnext_chain
                )
            else:
                llm_enhanced = self.enhance_with_llm(
                    random_enhanced, custom_llm_prompt, gemini_model
                )
            final_enhanced = llm_enhanced
        else:
            llm_enhanced = "LLM enhancement disabled"
            final_enhanced = random_enhanced
        
        return (final_enhanced, random_enhanced, llm_enhanced)
