# APNext H3 Prompt Preview
#
# Output node that shows a MiniMax-H3 prompt in a colour-coded, copyable
# panel. The rendering happens in web/js/h3_prompt_preview.js; this side just
# forwards the text to the UI and passes it through unchanged so it can be
# chained further.

from ...utils.constants import CUSTOM_CATEGORY


class H3PromptPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Shows a MiniMax-H3 prompt with colour-coded <Subject N>, <Picture N>, "
        "<Video N>, <Audio N>, <d> dialogue, [Shot N], speaker IDs and section "
        "headers, plus a one-click copy button. Passes the text through unchanged."
    )

    def preview(self, text):
        if isinstance(text, (list, tuple)):
            text = "\n\n".join(str(t) for t in text)
        text = "" if text is None else str(text)
        return {"ui": {"text": [text]}, "result": (text,)}
