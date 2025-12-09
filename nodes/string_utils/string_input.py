# StringInput Node

from ...utils.constants import CUSTOM_CATEGORY


class StringInput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "string": ("STRING", {"multiline": True, "forceInput": True}),
                "separator": ("STRING", {"default": " "}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output",)
    FUNCTION = "process"
    CATEGORY = CUSTOM_CATEGORY

    def process(self, prompt, string="", separator=" "):
        if string and prompt:
            result = f"{string}{separator}{prompt}"
        elif string:
            result = string
        else:
            result = prompt
        return (result,)

