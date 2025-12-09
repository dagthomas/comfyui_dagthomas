# RandomStringPicker Node

import random

from ...utils.constants import CUSTOM_CATEGORY


class RandomStringPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"multiline": True, "forceInput": True}),
                "string2": ("STRING", {"multiline": True, "forceInput": True}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "pick_random"
    CATEGORY = CUSTOM_CATEGORY

    def pick_random(self, string1, string2, seed=0):
        random.seed(seed)
        result = random.choice([string1, string2])
        return (result,)

