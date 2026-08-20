# APNext H3 Prompt Preview
#
# Output node that shows a MiniMax-H3 prompt in a colour-coded, copyable
# panel. The rendering happens in web/js/h3_prompt_preview.js; this side just
# forwards the text to the UI and passes it through unchanged so it can be
# chained further. Optional image_1..image_9 inputs are the reference images
# (<Picture N>); small thumbnails of them go to the UI so the panel can show
# them next to the tags (toggle "Thumbs" in the panel).

import os
import random

from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil
from .common import REFERENCE_IMAGE_NAMES

_THUMB_MAX = 192  # px, longest side


def _save_thumbnails(images, max_side=_THUMB_MAX):
    """Write downscaled PNGs to ComfyUI's temp folder; return UI entries."""
    try:
        import folder_paths
        out_dir = folder_paths.get_temp_directory()
    except Exception:
        return []
    os.makedirs(out_dir, exist_ok=True)
    prefix = "apnext_h3_prev_" + "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(6))
    thumbs = []
    for index, tensor in images:
        try:
            pil = tensor2pil(tensor[0] if tensor.dim() == 4 else tensor)
            pil = pil.convert("RGB")
            pil.thumbnail((max_side, max_side))
            name = f"{prefix}_{index}.png"
            pil.save(os.path.join(out_dir, name), compress_level=4)
            thumbs.append({
                "index": index,
                "filename": name,
                "subfolder": "",
                "type": "temp",
                "width": pil.width,
                "height": pil.height,
            })
        except Exception as exc:  # a bad image must not kill the preview
            print(f"⚠️ H3 Prompt Preview: thumbnail {index} failed: {exc}")
    return thumbs


class H3PromptPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "multiline": True}),
            },
            "optional": {
                name: ("IMAGE", {
                    "tooltip": (
                        f"Reference image {i} (<Picture {i}>). A thumbnail is shown in the "
                        "preview next to the tag; toggle with the Thumbs button in the panel."
                    ),
                })
                for i, name in enumerate(REFERENCE_IMAGE_NAMES, 1)
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "preview"
    OUTPUT_NODE = True
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "Shows a MiniMax-H3 prompt with colour-coded <Subject N>, <Picture N>, "
        "<Video N>, <Audio N>, <d> dialogue, [Shot N], speaker IDs and section "
        "headers, plus a one-click copy button and optional reference-image "
        "thumbnails (connect image_1..image_9). Passes the text through unchanged."
    )

    def preview(self, text, **image_slots):
        if isinstance(text, (list, tuple)):
            text = "\n\n".join(str(t) for t in text)
        text = "" if text is None else str(text)
        connected = [
            (slot, image_slots.get(name))
            for slot, name in enumerate(REFERENCE_IMAGE_NAMES, 1)
            if image_slots.get(name) is not None
        ]
        # number by CONNECTION ORDER, not slot index - the writers and the video
        # node label pictures <Picture 1>..<Picture N> the same way, so the
        # thumbnail lands on the tag even when the slots have gaps
        if connected and connected[-1][0] != len(connected):
            used = ", ".join(f"image_{slot}" for slot, _ in connected)
            print(
                f"⚠️ H3 Prompt Preview: reference images have a gap ({used}); numbering "
                f"thumbnails <Picture 1>..<Picture {len(connected)}> by connection order."
            )
        images = [(n, tensor) for n, (_slot, tensor) in enumerate(connected, 1)]
        thumbs = _save_thumbnails(images) if images else []
        return {"ui": {"text": [text], "thumbs": thumbs}, "result": (text,)}
