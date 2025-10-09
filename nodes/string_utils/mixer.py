# SentenceMixerNode Node

from ...utils.constants import CUSTOM_CATEGORY


class SentenceMixerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("STRING", {"multiline": True}),
            },
            "optional": {
                "input2": ("STRING", {"multiline": True}),
                "input3": ("STRING", {"multiline": True}),
                "input4": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "mix_sentences"
    CATEGORY = CUSTOM_CATEGORY

    def mix_sentences(self, input1, input2="", input3="", input4=""):
        def process_input(input_data):
            if isinstance(input_data, list):
                return " ".join(input_data)
            return input_data

        all_text = " ".join(
            filter(
                bool,
                [process_input(input) for input in [input1, input2, input3, input4]],
            )
        )

        sentences = []
        current_sentence = ""
        for char in all_text:
            current_sentence += char
            if char in [".", ","]:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())

        random.shuffle(sentences)

        result = " ".join(sentences)

        return (result,)
