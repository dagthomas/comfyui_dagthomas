# FileReaderNode Node

import json
import random

from ...utils.constants import CUSTOM_CATEGORY


class FileReaderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {"default": "./custom_nodes/comfyui_dagthomas/concat/output.json"},
                ),
                "amount": ("INT", {"default": 10, "min": 1, "max": 100}),
                "custom_tag": ("STRING", {"default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = CUSTOM_CATEGORY

    def generate_prompt(
        self, file_path: str, amount: int, custom_tag: str, seed: int = 0
    ) -> tuple:
        try:
            # Set the random seed if provided
            if seed != 0:
                random.seed(seed)

            # Step 1: Load JSON data from the file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as file:
                json_list = json.load(file)

            # Step 2: Randomly select the specified number of elements from the list
            random_values = random.sample(json_list, min(amount, len(json_list)))

            # Step 3: Join the selected elements into a single string separated by commas
            result_string = ", ".join(random_values)

            # Step 4: Add the custom tag if provided
            if custom_tag:
                result_string = f"{custom_tag}, {result_string}"

            return (result_string,)

        except Exception as e:
            return (f"Error: {str(e)}",)
