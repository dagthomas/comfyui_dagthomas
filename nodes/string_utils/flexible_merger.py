# FlexibleStringMergerNode Node

from ...utils.constants import CUSTOM_CATEGORY


class FlexibleStringMergerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
            },
            "optional": {
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = CUSTOM_CATEGORY

    def merge_strings(self, string1, string2="", string3="", string4=""):
        def process_input(s):
            if isinstance(s, list):
                return ", ".join(str(item) for item in s)
            return str(s).strip()

        strings = [
            process_input(s)
            for s in [string1, string2, string3, string4]
            if process_input(s)
        ]
        if not strings:
            return ""  # Return an empty string if no non-empty inputs
        return (" AND ".join(strings),)

