# APNext Sharpen Effect Node

import torch
import torch.nn.functional as F
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextSharpen:
    """
    APNext Sharpen Effect Node
    Sharpens images using unsharp mask technique for professional results
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1}),
                "radius": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
                "threshold": ("INT", {"default": 0, "min": 0, "max": 255, "step": 1}),
                "method": (["unsharp_mask", "high_pass", "edge_enhance"], {"default": "unsharp_mask"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_sharpen"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX"

    def apply_sharpen(self, images, strength, radius, threshold, method):
        """Apply sharpening effect to images using optimized tensor operations"""
        # Work directly with tensors for better performance
        device = images.device
        dtype = images.dtype
        
        # Process all images in batch
        result = self._apply_sharpening_method_tensor(
            images, strength, radius, threshold, method, device, dtype
        )
        
        return (result,)

    def _apply_sharpening_method_tensor(self, images, strength, radius, threshold, method, device, dtype):
        """Apply the specified sharpening method using tensor operations"""
        if method == "unsharp_mask":
            return self._unsharp_mask_tensor(images, strength, radius, threshold)
        elif method == "high_pass":
            return self._high_pass_sharpen_tensor(images, strength, radius)
        elif method == "edge_enhance":
            return self._edge_enhance_tensor(images, strength)
        
        return images

    def _unsharp_mask_tensor(self, images, strength, radius, threshold):
        """Apply unsharp mask sharpening using tensor operations"""
        # Create blurred version using gaussian blur
        blurred = self._gaussian_blur_tensor(images, radius)
        
        # Calculate the mask (difference between original and blurred)
        mask = images - blurred
        
        # Apply threshold if specified
        if threshold > 0:
            threshold_norm = threshold / 255.0
            threshold_mask = torch.abs(mask) > threshold_norm
            mask = torch.where(threshold_mask, mask, torch.zeros_like(mask))
        
        # Apply the unsharp mask
        sharpened = images + (mask * strength)
        
        # Clamp values to valid range
        return torch.clamp(sharpened, 0, 1)

    def _high_pass_sharpen_tensor(self, images, strength, radius):
        """Apply high-pass filter sharpening using tensor operations"""
        # Create heavily blurred version for high-pass
        blurred = self._gaussian_blur_tensor(images, radius * 2)
        
        # Create high-pass filter (original - blurred + 0.5)
        high_pass = images - blurred + 0.5
        high_pass = torch.clamp(high_pass, 0, 1)
        
        # Blend with original using overlay mode
        mask = images < 0.5
        result = torch.where(
            mask,
            2 * images * high_pass,
            1 - 2 * (1 - images) * (1 - high_pass)
        )
        
        # Apply strength
        result = images + (result - images) * strength
        return torch.clamp(result, 0, 1)

    def _edge_enhance_tensor(self, images, strength):
        """Apply edge enhancement sharpening using tensor operations"""
        # Create a simple sharpening kernel
        kernel = torch.tensor([
            [0, -1, 0],
            [-1, 5, -1],
            [0, -1, 0]
        ], dtype=images.dtype, device=images.device).view(1, 1, 3, 3)
        
        # Expand kernel for all channels
        kernel = kernel.expand(3, 1, 3, 3)
        
        # Apply convolution
        images_conv = images.permute(0, 3, 1, 2)  # [batch, channels, height, width]
        sharpened_conv = F.conv2d(images_conv, kernel, padding=1, groups=3)
        sharpened = sharpened_conv.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        
        # Blend with original
        result = images + (sharpened - images) * strength
        return torch.clamp(result, 0, 1)

    def _gaussian_blur_tensor(self, images, radius):
        """Apply gaussian blur using torch operations (shared with bloom)"""
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
