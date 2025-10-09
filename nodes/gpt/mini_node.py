# GPT Mini Node

import os
import random
from openai import OpenAI

from ...utils.constants import CUSTOM_CATEGORY, gpt_models


class GptMiniNode:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
                "gpt_model": (gpt_models, {"default": "gpt-4o-mini"}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
                "randomize_each_run": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
                "variation_instruction": (
                    "STRING", 
                    {"multiline": True, "default": "Generate different creative variations each time while maintaining the core concept."}
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def generate(
        self,
        input_text,
        happy_talk,
        compress,
        compression_level,
        poster,
        gpt_model="gpt-4o-mini",
        seed=-1,
        randomize_each_run=True,
        custom_base_prompt="",
        custom_title="",
        override="",
        variation_instruction="Generate different creative variations each time while maintaining the core concept.",
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
            print(f"🎲 GPT Mini Node using seed: {current_seed}")
            
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Including details about the film grain, color grading, and any artifacts or characteristics specific film and photography"""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            poster_prompt = f"""Analyze the provided description and extract key information to create a movie poster style description. Format the output as follows:

Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters.

Output should NEVER be markdown or include any formatting.
NEVER ADD ANY HGAPPY TALK IN THE PROMPT
Add how the title should look and state that its the main title.
Here is an example of a prompt, make the output like a movie poster: 
Title: Display the title "Verdant Spirits" in elegant and ethereal text, placed centrally at the top of the poster.

Main Character: Depict a serene and enchantingly beautiful woman with an aura of nature, her face partly adorned and encased with vibrant green foliage and delicate floral arrangements. She exudes an ethereal and mystical presence.

Background: The background should feature a dreamlike enchanted forest with lush greenery, vibrant flowers, and an ethereal glow emanating from the foliage. The scene should feel magical and otherworldly, suggesting a hidden world within nature.

Supporting Characters: Add an enigmatic skeletal figure entwined with glowing, bioluminescent leaves and plants, subtly blending with the dark, verdant background. This figure should evoke a sense of ancient wisdom and mysterious energy.

Studio Ghibli Branding: Incorporate the Studio Ghibli logo at the bottom center of the poster to establish it as an official Ghibli film.

Tagline: Include a tagline that reads: "Where Nature's Secrets Come to Life" prominently on the poster.

Visual Style: Ensure the overall visual style is consistent with Studio Ghibli s signature look   rich, detailed backgrounds, and characters imbued with a touch of whimsy and mystery. The colors should be lush and inviting, with an emphasis on the enchanting and mystical aspects of nature."""

            if poster:
                base_prompt = poster_prompt
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = (
                    default_happy_prompt if happy_talk else default_simple_prompt
                )

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100,
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            # Add variation instruction to encourage different outputs each time
            if randomize_each_run and variation_instruction.strip():
                base_prompt += f"\n\nIMPORTANT: {variation_instruction} Use random seed {current_seed} to ensure unique creative variations."

            # Add the override at the beginning of the prompt if provided
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            response = self.client.chat.completions.create(
                model=gpt_model,
                messages=[
                    {
                        "role": "user",
                        "content": f"{final_prompt}\nDescription: {input_text}",
                    }
                ],
                # Add some randomness to the API call itself
                temperature=0.9 if randomize_each_run else 0.7,
                top_p=0.95 if randomize_each_run else 0.8,
                seed=current_seed if not randomize_each_run else None,  # Only use seed if not randomizing
            )

            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)
