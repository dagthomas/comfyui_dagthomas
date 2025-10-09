# GPT Vision Node

import os
import base64
import numpy as np
from io import BytesIO
from PIL import Image
from openai import OpenAI

from ...utils.constants import CUSTOM_CATEGORY, gpt_models


class GptVisionNode:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
                "gpt_model": (gpt_models,),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def tensor_to_pil(self, img_tensor):
        i = 255.0 * img_tensor.cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def analyze_images(
        self,
        images,
        happy_talk,
        compress,
        compression_level,
        poster,
        gpt_model="gpt-5",
        custom_base_prompt="",
        custom_title="",
        override="",
    ):
        try:
            default_happy_prompt = """Analyze the provided images and create a detailed visually descriptive caption that combines elements from all images into a single cohesive composition.Imagine all images being movie stills from real movies. This caption will be used as a prompt for a text-to-image AI system. Focus on:
1. Detailed visual descriptions of characters, including ethnicity, skin tone, expressions, etc.
2. Overall scene and background details.
3. Image style, photographic techniques, direction of photo taken.
4. Cinematography aspects with technical details.
5. If multiple characters are present, describe an interesting interaction between two primary characters.
6. Incorporate a specific movie director's visual style (e.g., Wes Anderson, Christopher Nolan, Quentin Tarantino).
7. Describe the lighting setup in detail, including type, color, and placement of light sources.

If you want to add text, state the font, style and where it should be placed, a sign, a poster, etc. The text should be captioned like this " ".
Always describn how characters look at each other, their expressions, and the overall mood of the scene.
Examples of prompts to generate: 

1. Ethereal cyborg woman, bioluminescent jellyfish headdress. Steampunk goggles blend with translucent tentacles. Cracked porcelain skin meets iridescent scales. Mechanical implants and delicate tendrils intertwine. Human features with otherworldly glow. Dreamy aquatic hues contrast weathered metal. Reflective eyes capture unseen worlds. Soft bioluminescence meets harsh desert backdrop. Fusion of organic and synthetic, ancient and futuristic. Hyper-detailed textures, surreal atmosphere.

2. Photo of a broken ruined cyborg girl in a landfill, robot, body is broken with scares and holes,half the face is android,laying on the ground, creating a hyperpunk scene with desaturated dark red and blue details, colorful polaroid with vibrant colors, (vacations, high resolution:1.3), (small, selective focus, european film:1.2)

3. Horror-themed (extreme close shot of eyes :1.3) of nordic woman, (war face paint:1.2), mohawk blonde haircut wit thin braids, runes tattoos, sweat, (detailed dirty skin:1.3) shiny, (epic battleground backgroun :1.2), . analog, haze, ( lens blur :1.3) , hard light, sharp focus on eyes, low saturation

ALWAYS remember to out that it is a cinematic movie still and describe the film grain, color grading, and any artifacts or characteristics specific to film photography.
ALWAYS create the output as one scene, never transition between scenes.
"""

            default_simple_prompt = """Analyze the provided images and create a brief, straightforward caption that combines key elements from all images. Focus on the main subjects, overall scene, and atmosphere. Provide a clear and concise description in one or two sentences, suitable for a text-to-image AI system."""

            poster_prompt = f"""Analyze the provided images and extract key information to create a cinamtic movie poster style description. Format the output as follows:

Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters.

Here is an example of a prompt: 
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

            # Add the override at the beginning of the prompt
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            messages = [
                {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
            ]

            # Process each image in the batch
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                base64_image = self.encode_image(pil_image)
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

            response = self.client.chat.completions.create(
                model=gpt_model, messages=messages
            )
            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Images tensor shape: {images.shape}")
            print(f"Images tensor type: {images.dtype}")
            return (f"Error occurred while processing the request: {str(e)}",)
