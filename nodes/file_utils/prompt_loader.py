# CustomPromptLoader Node

import os
import chardet
from ...utils.constants import CUSTOM_CATEGORY, prompt_dir


class CustomPromptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_file": (s.get_prompt_files(),),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_prompt"
    CATEGORY = CUSTOM_CATEGORY

    @staticmethod
    def get_prompt_files():
        return [f for f in os.listdir(prompt_dir) if f.endswith(".txt")]

    @classmethod
    def IS_CHANGED(s, prompt_file):
        return float("nan")  # This ensures the widget is always refreshed

    def load_prompt(self, prompt_file):
        file_path = os.path.join(prompt_dir, prompt_file)

        # Detect the file encoding
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"]

        # Read the file with the detected encoding
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()
        except UnicodeDecodeError:
            # If the detected encoding fails, try UTF-8 as a fallback
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                # If UTF-8 also fails, try ANSI (Windows-1252) as a last resort
                with open(file_path, "r", encoding="windows-1252") as file:
                    content = file.read()

        return (content,)
