# Gemini Text Only Node

import os
import re
import google.generativeai as genai

from ...utils.constants import CUSTOM_CATEGORY, gemini_models


class GeminiTextOnly:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=self.gemini_api_key)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "gemini_model": (gemini_models,),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "clip_l")
    FUNCTION = "process_text"
    CATEGORY = CUSTOM_CATEGORY

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def process_text(
        self,
        custom_prompt="",
        additive_prompt="",
        tag="ohwx man",
        sex="male",
        pronouns="him, his",
        dynamic_prompt=False,
        words="100",
        gemini_model="gemini-flash-latest",
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Generate a response."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

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

            result = response.text

            return (
                result,
                self.extract_first_two_sentences(result),
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            return (error_message, error_message[:100])
