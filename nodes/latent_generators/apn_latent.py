# APNLatent Node

import nodes
from ...utils.constants import CUSTOM_CATEGORY
import torch


class APNLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "megapixel_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
                "aspect_ratio": (
                    ["1:1", "3:2", "4:3", "16:9", "21:9"],
                    {"default": "1:1"},
                ),
                "is_portrait": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def generate(
        self,
        width=1024,
        height=1024,
        batch_size=1,
        megapixel_scale=1.0,
        aspect_ratio="1:1",
        is_portrait=False,
    ):
        def adjust_dimensions(megapixels, aspect_ratio, is_portrait):
            aspect_ratios = {
                "1:1": (1, 1),
                "3:2": (3, 2),
                "4:3": (4, 3),
                "16:9": (16, 9),
                "21:9": (21, 9),
            }

            ar_width, ar_height = aspect_ratios[aspect_ratio]
            if is_portrait:
                ar_width, ar_height = ar_height, ar_width

            total_pixels = int(megapixels * 1_000_000)

            width = int((total_pixels * ar_width / ar_height) ** 0.5)
            height = int(total_pixels / width)

            # Round to nearest multiple of 64
            width = (width + 32) // 64 * 64
            height = (height + 32) // 64 * 64

            return width, height

        if width == 0 or height == 0:
            width, height = adjust_dimensions(
                megapixel_scale, aspect_ratio, is_portrait
            )
        else:
            # If width and height are provided, adjust them to fit within the megapixel scale
            current_mp = (width * height) / 1_000_000
            if current_mp > megapixel_scale:
                scale_factor = (megapixel_scale / current_mp) ** 0.5
                width = int((width * scale_factor + 32) // 64 * 64)
                height = int((height * scale_factor + 32) // 64 * 64)

            # Swap width and height if portrait is selected and current orientation doesn't match
            if is_portrait and width > height:
                width, height = height, width
            elif not is_portrait and height > width:
                width, height = height, width

        latent = (
            torch.ones([batch_size, 16, height // 8, width // 8], device=self.device)
            * 0.0609
        )
        return ({"samples": latent}, width, height)

