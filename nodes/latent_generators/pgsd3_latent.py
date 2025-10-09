# PGSD3LatentGenerator Node

import nodes
from ...utils.constants import CUSTOM_CATEGORY
import torch


class PGSD3LatentGenerator:
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def generate(self, width=1024, height=1024, batch_size=1):
        def adjust_dimensions(width, height):
            megapixel = 1_000_000
            multiples = 64

            if width == 0 and height != 0:
                height = int(height)
                width = megapixel // height

                if width % multiples != 0:
                    width += multiples - (width % multiples)

                if width * height > megapixel:
                    width -= multiples

            elif height == 0 and width != 0:
                width = int(width)
                height = megapixel // width

                if height % multiples != 0:
                    height += multiples - (height % multiples)

                if width * height > megapixel:
                    height -= multiples

            elif width == 0 and height == 0:
                width = 1024
                height = 1024

            return width, height

        width, height = adjust_dimensions(width, height)

        latent = (
            torch.ones([batch_size, 16, height // 8, width // 8], device=self.device)
            * 0.0609
        )
        return ({"samples": latent},)

