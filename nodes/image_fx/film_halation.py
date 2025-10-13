# APNext Film Halation Effect Node

import torch
import torch.nn.functional as F
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextFilmHalation:
    """
    APNext Film Halation Effect Node
    Simulates film halation - light bleeding around bright objects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "threshold": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01}),
                "radius": ("FLOAT", {"default": 20.0, "min": 1.0, "max": 100.0, "step": 0.5}),
                "color_tint": (["none", "warm", "cool", "red", "blue", "custom"], {"default": "warm"}),
                "custom_color_r": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "custom_color_g": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 2.0, "step": 0.01}),
                "custom_color_b": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.01}),
                "falloff": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 4.0, "step": 0.1}),
                "film_type": (["color_negative", "slide_film", "vintage", "modern"], {"default": "color_negative"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_film_halation"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_film_halation(self, images, intensity, threshold, radius, color_tint, 
                           custom_color_r, custom_color_g, custom_color_b, falloff, film_type):
        """Apply film halation effect to images"""
        device = images.device
        dtype = images.dtype
        
        # Apply halation effect
        result = self._create_halation_effect(
            images, intensity, threshold, radius, color_tint,
            custom_color_r, custom_color_g, custom_color_b, falloff, film_type
        )
        
        return (result,)

    def _create_halation_effect(self, images, intensity, threshold, radius, color_tint,
                               custom_color_r, custom_color_g, custom_color_b, falloff, film_type):
        """Create film halation effect using tensor operations"""
        batch_size, height, width, channels = images.shape
        device = images.device
        dtype = images.dtype
        
        # Calculate luminance for threshold detection
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        
        # Create mask for bright areas
        bright_mask = (luminance > threshold).float()
        
        # Extract bright areas
        bright_areas = images * bright_mask.unsqueeze(-1)
        
        # Apply film-specific characteristics
        bright_areas = self._apply_film_characteristics(bright_areas, film_type)
        
        # Create halation glow
        halation_glow = self._create_glow_effect(bright_areas, radius, falloff)
        
        # Apply color tinting
        tinted_glow = self._apply_color_tint(
            halation_glow, color_tint, custom_color_r, custom_color_g, custom_color_b
        )
        
        # Blend halation with original image
        result = self._blend_halation(images, tinted_glow, intensity)
        
        return torch.clamp(result, 0, 1)

    def _apply_film_characteristics(self, bright_areas, film_type):
        """Apply film-specific characteristics to bright areas"""
        if film_type == "color_negative":
            # Color negative film has strong red/orange halation
            bright_areas[:, :, :, 0] *= 1.3  # Enhance red
            bright_areas[:, :, :, 1] *= 1.1  # Slightly enhance green
            bright_areas[:, :, :, 2] *= 0.8  # Reduce blue
            
        elif film_type == "slide_film":
            # Slide film has more neutral halation with slight magenta cast
            bright_areas[:, :, :, 0] *= 1.1  # Slight red enhancement
            bright_areas[:, :, :, 1] *= 0.95  # Slight green reduction
            bright_areas[:, :, :, 2] *= 1.05  # Slight blue enhancement
            
        elif film_type == "vintage":
            # Vintage film has warm, yellowish halation
            bright_areas[:, :, :, 0] *= 1.4  # Strong red
            bright_areas[:, :, :, 1] *= 1.2  # Enhanced green (yellow)
            bright_areas[:, :, :, 2] *= 0.6  # Reduced blue
            
        elif film_type == "modern":
            # Modern film has more controlled, neutral halation
            bright_areas *= 1.1  # Slight overall enhancement
        
        return torch.clamp(bright_areas, 0, 2)  # Allow overexposure

    def _create_glow_effect(self, bright_areas, radius, falloff):
        """Create glow effect using multiple gaussian blurs"""
        # Create multiple blur layers for realistic halation
        glow_layers = []
        
        # Primary glow (tight)
        primary_glow = self._gaussian_blur_tensor(bright_areas, radius * 0.3)
        glow_layers.append(primary_glow * 0.6)
        
        # Secondary glow (medium)
        secondary_glow = self._gaussian_blur_tensor(bright_areas, radius * 0.7)
        glow_layers.append(secondary_glow * 0.3)
        
        # Tertiary glow (wide)
        tertiary_glow = self._gaussian_blur_tensor(bright_areas, radius)
        glow_layers.append(tertiary_glow * 0.1)
        
        # Combine glow layers
        combined_glow = sum(glow_layers)
        
        # Apply falloff
        if falloff != 1.0:
            # Create distance-based falloff
            luminance = 0.299 * combined_glow[:, :, :, 0] + 0.587 * combined_glow[:, :, :, 1] + 0.114 * combined_glow[:, :, :, 2]
            falloff_factor = torch.pow(luminance.unsqueeze(-1), 1.0 / falloff)
            combined_glow = combined_glow * falloff_factor
        
        return combined_glow

    def _apply_color_tint(self, glow, color_tint, custom_r, custom_g, custom_b):
        """Apply color tinting to the halation glow"""
        if color_tint == "none":
            return glow
        
        tint_colors = {
            "warm": torch.tensor([1.2, 1.0, 0.8], device=glow.device, dtype=glow.dtype),
            "cool": torch.tensor([0.8, 1.0, 1.2], device=glow.device, dtype=glow.dtype),
            "red": torch.tensor([1.5, 0.8, 0.8], device=glow.device, dtype=glow.dtype),
            "blue": torch.tensor([0.8, 0.9, 1.4], device=glow.device, dtype=glow.dtype),
            "custom": torch.tensor([custom_r, custom_g, custom_b], device=glow.device, dtype=glow.dtype)
        }
        
        if color_tint in tint_colors:
            tint = tint_colors[color_tint].view(1, 1, 1, 3)
            tinted_glow = glow * tint
        else:
            tinted_glow = glow
        
        return tinted_glow

    def _blend_halation(self, original, halation, intensity):
        """Blend halation effect with original image"""
        # Use screen blend mode for realistic halation
        # Screen formula: 1 - (1 - base) * (1 - overlay)
        halation_scaled = halation * intensity
        
        # Apply screen blending
        result = 1 - (1 - original) * (1 - halation_scaled)
        
        return result

    def _gaussian_blur_tensor(self, images, radius):
        """Apply gaussian blur using torch operations"""
        if radius < 0.5:
            return images
            
        # Convert radius to sigma
        sigma = radius / 3.0
        
        # Create gaussian kernel
        kernel_size = int(2 * radius + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Limit kernel size for performance
        kernel_size = min(kernel_size, 99)
        
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
