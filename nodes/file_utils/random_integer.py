# RandomIntegerNode Node

from ...utils.constants import CUSTOM_CATEGORY
import random


class RandomIntegerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": (
                    "INT",
                    {"default": 0, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
                "max_value": (
                    "INT",
                    {"default": 10, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_random_int"
    CATEGORY = CUSTOM_CATEGORY  # Replace with your actual category

    def generate_random_int(self, min_value, max_value):
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        result = random.randint(min_value, max_value)
        # print(f"Generating random int between {min_value} and {max_value}: {result}")
        return (result,)

