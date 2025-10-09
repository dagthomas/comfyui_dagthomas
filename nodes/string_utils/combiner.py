# Dynamic String Combiner Node

from ...utils.constants import CUSTOM_CATEGORY


class DynamicStringCombinerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_inputs": (["1", "2", "3", "4", "5"],),
                "user_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "string1": ("STRING", {"multiline": False}),
                "string2": ("STRING", {"multiline": False}),
                "string3": ("STRING", {"multiline": False}),
                "string4": ("STRING", {"multiline": False}),
                "string5": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine_strings"
    CATEGORY = CUSTOM_CATEGORY

    def combine_strings(
        self,
        num_inputs,
        user_text,
        string1="",
        string2="",
        string3="",
        string4="",
        string5="",
    ):
        # Convert num_inputs to integer
        n = int(num_inputs)

        # Get the specified number of input strings
        input_strings = [string1, string2, string3, string4, string5][:n]

        # Combine the input strings
        combined = ", ".join(s for s in input_strings if s.strip())

        # Append the user_text to the result
        result = f"{combined}\nUser Input: {user_text}"

        return (result,)

    @classmethod
    def IS_CHANGED(
        s, num_inputs, user_text, string1, string2, string3, string4, string5
    ):
        return float(num_inputs)
