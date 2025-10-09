# StringMergerNode Node

from ...utils.constants import CUSTOM_CATEGORY


class StringMergerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": "", "forceInput": True}),
                "string2": ("STRING", {"default": "", "forceInput": True}),
                "use_and": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = CUSTOM_CATEGORY

    def merge_strings(self, string1, string2, use_and):
        def process_input(s):
            if isinstance(s, list):
                return ",".join(str(item).strip() for item in s)
            return str(s).strip()

        processed_string1 = process_input(string1)
        processed_string2 = process_input(string2)
        separator = " AND " if use_and else ","
        merged = f"{processed_string1}{separator}{processed_string2}"

        # Remove double commas and clean spaces around commas
        merged = merged.replace(",,", ",").replace(" ,", ",").replace(", ", ",")

        # Clean leading and trailing spaces
        merged = merged.strip()

        return (merged,)
