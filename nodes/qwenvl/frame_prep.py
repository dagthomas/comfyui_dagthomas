# QwenVL Frame Prep Node
# Prepares multiple images for QwenVL Next Scene node

import numpy as np
import torch
from PIL import Image

from ...utils.constants import CUSTOM_CATEGORY


class QwenVLFramePrep:
    """
    Prepares multiple images for the QwenVL Next Scene node.
    - Accepts multiple image inputs
    - Scales images to max width (default 1024) maintaining aspect ratio
    - Batches them into a single IMAGE output for QwenVL Next Scene
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "max_width": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "max_height": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
            },
            "optional": {
                "image_1": ("IMAGE",),
                "image_2": ("IMAGE",),
                "image_3": ("IMAGE",),
                "image_4": ("IMAGE",),
                "image_5": ("IMAGE",),
                "image_batch": ("IMAGE",),  # Can also accept a pre-batched input
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("images", "frame_count")
    FUNCTION = "prepare_frames"
    CATEGORY = f"{CUSTOM_CATEGORY}/LLM"

    @staticmethod
    def tensor2pil(image):
        """Convert tensor to PIL image"""
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        """Convert PIL image to tensor"""
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0)

    def scale_image(self, pil_image, max_width, max_height):
        """Scale image to fit within max dimensions while maintaining aspect ratio"""
        width, height = pil_image.size
        
        # Calculate scale factors
        scale_w = max_width / width if width > max_width else 1.0
        scale_h = max_height / height if height > max_height else 1.0
        scale = min(scale_w, scale_h)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            pil_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        return pil_image

    def prepare_frames(
        self,
        max_width=1024,
        max_height=1024,
        image_1=None,
        image_2=None,
        image_3=None,
        image_4=None,
        image_5=None,
        image_batch=None,
    ):
        frames = []
        
        # Collect individual images
        individual_images = [image_1, image_2, image_3, image_4, image_5]
        
        for img in individual_images:
            if img is not None:
                # Handle batched inputs - take first frame from each
                if len(img.shape) == 4:
                    frames.append(img[0])
                else:
                    frames.append(img)
        
        # Add frames from batch input
        if image_batch is not None:
            if len(image_batch.shape) == 4:
                for i in range(image_batch.shape[0]):
                    frames.append(image_batch[i])
            else:
                frames.append(image_batch)
        
        if not frames:
            raise ValueError("No images provided! Connect at least one image input.")
        
        # Process and scale frames
        processed_frames = []
        target_size = None
        
        for i, frame_tensor in enumerate(frames):
            # Convert to PIL
            pil_image = self.tensor2pil(frame_tensor)
            
            # Scale to max dimensions
            pil_image = self.scale_image(pil_image, max_width, max_height)
            
            # Use first frame's size as target for all frames
            if target_size is None:
                target_size = pil_image.size
            else:
                # Resize to match first frame if different
                if pil_image.size != target_size:
                    pil_image = pil_image.resize(target_size, Image.Resampling.LANCZOS)
            
            # Convert back to tensor
            tensor = self.pil2tensor(pil_image)
            processed_frames.append(tensor)
            
            print(f"🖼️ Frame {i+1}: {pil_image.size[0]}x{pil_image.size[1]}")
        
        # Stack into batch
        output = torch.stack(processed_frames, dim=0)
        frame_count = len(processed_frames)
        
        print(f"✅ Prepared {frame_count} frames for QwenVL Next Scene")
        print(f"   Output shape: {output.shape}")
        
        return (output, frame_count)
