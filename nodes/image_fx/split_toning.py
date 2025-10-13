# APNext Split Toning Effect Node

import torch
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextSplitToning:
    """
    APNext Split Toning Effect Node
    Applies different colors to highlights and shadows independently
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                # Highlight controls
                "highlight_hue": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "highlight_saturation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "highlight_luminance": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Shadow controls
                "shadow_hue": ("FLOAT", {"default": 240.0, "min": 0.0, "max": 360.0, "step": 1.0}),
                "shadow_saturation": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_luminance": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                # Balance and blending
                "balance": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "midtone_contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["color", "soft_light", "overlay", "multiply"], {"default": "color"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_split_toning"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_split_toning(self, images, highlight_hue, highlight_saturation, highlight_luminance,
                          shadow_hue, shadow_saturation, shadow_luminance,
                          balance, midtone_contrast, blend_mode, opacity):
        """Apply split toning effect to images"""
        device = images.device
        dtype = images.dtype
        
        # Apply split toning
        result = self._apply_split_toning_tensor(
            images, highlight_hue, highlight_saturation, highlight_luminance,
            shadow_hue, shadow_saturation, shadow_luminance,
            balance, midtone_contrast, blend_mode, opacity
        )
        
        return (result,)

    def _apply_split_toning_tensor(self, images, h_hue, h_sat, h_lum, s_hue, s_sat, s_lum,
                                  balance, midtone_contrast, blend_mode, opacity):
        """Apply split toning using tensor operations"""
        batch_size, height, width, channels = images.shape
        device = images.device
        dtype = images.dtype
        
        # Convert to HSV for easier manipulation
        hsv_images = self._rgb_to_hsv_tensor(images)
        
        # Extract luminance for masking
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        luminance = luminance.unsqueeze(-1)
        
        # Adjust luminance threshold based on balance
        adjusted_h_lum = h_lum + balance * 0.2
        adjusted_s_lum = s_lum - balance * 0.2
        adjusted_h_lum = torch.clamp(torch.tensor(adjusted_h_lum), 0.0, 1.0)
        adjusted_s_lum = torch.clamp(torch.tensor(adjusted_s_lum), 0.0, 1.0)
        
        # Create smooth masks for highlights and shadows
        highlight_mask = self._create_luminance_mask(luminance, adjusted_h_lum, softness=0.2)
        shadow_mask = self._create_luminance_mask(luminance, adjusted_s_lum, invert=True, softness=0.2)
        
        # Create toned colors
        highlight_color = self._create_toned_color(h_hue, h_sat, device, dtype)
        shadow_color = self._create_toned_color(s_hue, s_sat, device, dtype)
        
        # Apply toning based on blend mode
        if blend_mode == "color":
            result = self._apply_color_blend(images, highlight_color, shadow_color, 
                                           highlight_mask, shadow_mask)
        elif blend_mode == "soft_light":
            result = self._apply_soft_light_blend(images, highlight_color, shadow_color,
                                                highlight_mask, shadow_mask)
        elif blend_mode == "overlay":
            result = self._apply_overlay_blend(images, highlight_color, shadow_color,
                                             highlight_mask, shadow_mask)
        elif blend_mode == "multiply":
            result = self._apply_multiply_blend(images, highlight_color, shadow_color,
                                              highlight_mask, shadow_mask)
        
        # Apply midtone contrast
        if midtone_contrast != 0.0:
            result = self._apply_midtone_contrast_tensor(result, midtone_contrast)
        
        # Blend with original based on opacity
        result = images * (1 - opacity) + result * opacity
        
        # Clamp values
        result = torch.clamp(result, 0, 1)
        return result

    def _create_luminance_mask(self, luminance, threshold, invert=False, softness=0.1):
        """Create smooth luminance-based mask"""
        if invert:
            mask = torch.sigmoid((threshold - luminance) / softness)
        else:
            mask = torch.sigmoid((luminance - threshold) / softness)
        return mask

    def _create_toned_color(self, hue, saturation, device, dtype):
        """Create a color from hue and saturation"""
        # Convert hue from degrees to 0-1 range
        h = (hue % 360.0) / 360.0
        s = saturation
        v = 1.0  # Full brightness
        
        # Convert HSV to RGB
        rgb = self._hsv_to_rgb_single(h, s, v, device, dtype)
        return rgb

    def _hsv_to_rgb_single(self, h, s, v, device, dtype):
        """Convert single HSV values to RGB"""
        h = torch.tensor(h, device=device, dtype=dtype)
        s = torch.tensor(s, device=device, dtype=dtype)
        v = torch.tensor(v, device=device, dtype=dtype)
        
        c = v * s
        x = c * (1 - torch.abs((h * 6) % 2 - 1))
        m = v - c
        
        h_i = torch.floor(h * 6).long()
        
        # Create RGB based on hue sector
        if h_i == 0:
            r, g, b = c, x, 0
        elif h_i == 1:
            r, g, b = x, c, 0
        elif h_i == 2:
            r, g, b = 0, c, x
        elif h_i == 3:
            r, g, b = 0, x, c
        elif h_i == 4:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return torch.stack([r + m, g + m, b + m])

    def _apply_color_blend(self, images, highlight_color, shadow_color, h_mask, s_mask):
        """Apply color blend mode for split toning"""
        # Convert to HSV
        hsv = self._rgb_to_hsv_tensor(images)
        
        # Replace hue and saturation while preserving luminance
        h_color_expanded = highlight_color.view(1, 1, 1, 3).expand_as(images)
        s_color_expanded = shadow_color.view(1, 1, 1, 3).expand_as(images)
        
        # Blend colors
        toned = images.clone()
        toned = toned * (1 - h_mask) + h_color_expanded * h_mask
        toned = toned * (1 - s_mask) + s_color_expanded * s_mask
        
        return toned

    def _apply_soft_light_blend(self, images, highlight_color, shadow_color, h_mask, s_mask):
        """Apply soft light blend mode"""
        h_color_expanded = highlight_color.view(1, 1, 1, 3).expand_as(images)
        s_color_expanded = shadow_color.view(1, 1, 1, 3).expand_as(images)
        
        # Soft light formula
        h_blend = torch.where(
            h_color_expanded <= 0.5,
            images - (1 - 2 * h_color_expanded) * images * (1 - images),
            images + (2 * h_color_expanded - 1) * (torch.sqrt(images) - images)
        )
        
        s_blend = torch.where(
            s_color_expanded <= 0.5,
            images - (1 - 2 * s_color_expanded) * images * (1 - images),
            images + (2 * s_color_expanded - 1) * (torch.sqrt(images) - images)
        )
        
        result = images * (1 - h_mask) + h_blend * h_mask
        result = result * (1 - s_mask) + s_blend * s_mask
        
        return result

    def _apply_overlay_blend(self, images, highlight_color, shadow_color, h_mask, s_mask):
        """Apply overlay blend mode"""
        h_color_expanded = highlight_color.view(1, 1, 1, 3).expand_as(images)
        s_color_expanded = shadow_color.view(1, 1, 1, 3).expand_as(images)
        
        # Overlay formula
        h_blend = torch.where(
            images <= 0.5,
            2 * images * h_color_expanded,
            1 - 2 * (1 - images) * (1 - h_color_expanded)
        )
        
        s_blend = torch.where(
            images <= 0.5,
            2 * images * s_color_expanded,
            1 - 2 * (1 - images) * (1 - s_color_expanded)
        )
        
        result = images * (1 - h_mask) + h_blend * h_mask
        result = result * (1 - s_mask) + s_blend * s_mask
        
        return result

    def _apply_multiply_blend(self, images, highlight_color, shadow_color, h_mask, s_mask):
        """Apply multiply blend mode"""
        h_color_expanded = highlight_color.view(1, 1, 1, 3).expand_as(images)
        s_color_expanded = shadow_color.view(1, 1, 1, 3).expand_as(images)
        
        h_blend = images * h_color_expanded
        s_blend = images * s_color_expanded
        
        result = images * (1 - h_mask) + h_blend * h_mask
        result = result * (1 - s_mask) + s_blend * s_mask
        
        return result

    def _apply_midtone_contrast_tensor(self, images, contrast):
        """Apply contrast specifically to midtones"""
        # Create midtone mask
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        luminance = luminance.unsqueeze(-1)
        
        # Gaussian-like curve centered at 0.5
        midtone_mask = torch.exp(-((luminance - 0.5) ** 2) / (2 * 0.2 ** 2))
        
        # Apply contrast
        contrast_factor = 1.0 + contrast
        mid_point = 0.5
        contrasted = (images - mid_point) * contrast_factor + mid_point
        
        # Blend based on midtone mask
        result = images * (1 - midtone_mask) + contrasted * midtone_mask
        return result

    def _rgb_to_hsv_tensor(self, rgb):
        """Convert RGB to HSV color space"""
        r, g, b = rgb[:, :, :, 0], rgb[:, :, :, 1], rgb[:, :, :, 2]
        
        max_val, max_idx = torch.max(rgb, dim=3)
        min_val, _ = torch.min(rgb, dim=3)
        
        delta = max_val - min_val
        
        # Hue calculation
        hue = torch.zeros_like(max_val)
        mask = delta != 0
        
        # Red is max
        red_max = (max_idx == 0) & mask
        hue[red_max] = ((g[red_max] - b[red_max]) / delta[red_max]) % 6
        
        # Green is max
        green_max = (max_idx == 1) & mask
        hue[green_max] = (b[green_max] - r[green_max]) / delta[green_max] + 2
        
        # Blue is max
        blue_max = (max_idx == 2) & mask
        hue[blue_max] = (r[blue_max] - g[blue_max]) / delta[blue_max] + 4
        
        hue = hue / 6.0
        
        # Saturation
        saturation = torch.zeros_like(max_val)
        saturation[max_val != 0] = delta[max_val != 0] / max_val[max_val != 0]
        
        # Value
        value = max_val
        
        return torch.stack([hue, saturation, value], dim=3)
