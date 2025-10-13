# APNext Bloom Effect Node

import torch
import torch.nn.functional as F
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextBloom:
    """
    APNext Bloom Effect Node
    Creates a bloom effect by making bright areas glow using gaussian blur and blend modes
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("FLOAT", {"default": 15.0, "min": 1.0, "max": 50.0, "step": 0.5}),
                "blend_mode": (["additive", "screen", "overlay"], {"default": "additive"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_bloom"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX"

    def apply_bloom(self, images, intensity, threshold, blur_radius, blend_mode):
        """Apply bloom effect to images using optimized tensor operations"""
        # Work directly with tensors for better performance
        batch_size = images.shape[0]
        device = images.device
        dtype = images.dtype
        
        # Process all images in batch
        result = self._create_bloom_effect_tensor(
            images, intensity, threshold, blur_radius, blend_mode, device, dtype
        )
        
        return (result,)

    def _create_bloom_effect_tensor(self, images, intensity, threshold, blur_radius, blend_mode, device, dtype):
        """Create bloom effect using tensor operations for better performance"""
        # Apply threshold to isolate bright areas
        bloom_layer = self._apply_threshold_tensor(images, threshold)
        
        # Apply gaussian blur using torch operations
        bloom_layer = self._gaussian_blur_tensor(bloom_layer, blur_radius)
        
        # Blend the bloom layer with the original image
        result = self._blend_images_tensor(images, bloom_layer, blend_mode, intensity)
        
        return result

    def _apply_threshold_tensor(self, images, threshold):
        """Apply threshold to isolate bright areas using tensor operations"""
        # Calculate luminance using tensor operations
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        
        # Create mask for bright areas
        bright_mask = luminance > threshold
        
        # Apply threshold - keep only pixels above threshold
        result = images.clone()
        bright_mask = bright_mask.unsqueeze(-1).expand_as(images)
        result = torch.where(bright_mask, images, torch.zeros_like(images))
        
        return result

    def _blend_images_tensor(self, base, overlay, blend_mode, intensity):
        """Blend two images using tensor operations"""
        overlay = overlay * intensity
        
        if blend_mode == "additive":
            result = base + overlay
        elif blend_mode == "screen":
            result = 1 - (1 - base) * (1 - overlay)
        elif blend_mode == "overlay":
            mask = base < 0.5
            result = torch.where(
                mask,
                2 * base * overlay,
                1 - 2 * (1 - base) * (1 - overlay)
            )
        
        # Clamp values to [0, 1]
        result = torch.clamp(result, 0, 1)
        return result
    
    def _gaussian_blur_tensor(self, images, radius):
        """Apply gaussian blur using torch operations"""
        # Convert radius to sigma (approximation)
        sigma = radius / 3.0
        
        # Create gaussian kernel
        kernel_size = int(2 * radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
            
        # Create 1D gaussian kernel
        x = torch.arange(kernel_size, dtype=images.dtype, device=images.device)
        x = x - kernel_size // 2
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        # Apply separable gaussian blur
        # Reshape for conv operations: [batch, channels, height, width]
        images_conv = images.permute(0, 3, 1, 2)
        
        # Horizontal blur
        kernel_h = kernel_1d.view(1, 1, 1, -1).expand(3, 1, 1, -1)
        blurred_h = F.conv2d(images_conv, kernel_h, padding=(0, kernel_size//2), groups=3)
        
        # Vertical blur
        kernel_v = kernel_1d.view(1, 1, -1, 1).expand(3, 1, -1, 1)
        blurred = F.conv2d(blurred_h, kernel_v, padding=(kernel_size//2, 0), groups=3)
        
        # Reshape back to original format: [batch, height, width, channels]
        return blurred.permute(0, 2, 3, 1)
