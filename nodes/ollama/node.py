# OllamaNode Node

from ...utils.constants import CUSTOM_CATEGORY
import json
import requests


class OllamaNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
                "poster": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_model": ("STRING", {"default": "llama3.1:8b"}),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434/api/generate"},
                ),
                "custom_title": ("STRING", {"default": ""}),  # New custom title field
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),  # New override field
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
        custom_base_prompt="",
        custom_model="llama3.1:8b",
        ollama_url="http://localhost:11434/api/generate",
        custom_title="",
        override="",
    ):
        try:
            default_happy_prompt = """Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:
If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style
Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus
Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 150 WORDS"""

            default_simple_prompt = """Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:
If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style
Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus
Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 200 WORDS"""  ##

            poster_prompt = f"""
Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:

If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style

Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus

Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 75 WORDS"""

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

            # Add the override at the beginning of the prompt if provided
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            prompt = f"{final_prompt}\nDescription: {input_text}"

            payload = {"model": custom_model, "prompt": prompt, "stream": False}

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            return (result,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)
