# APNext Cross Processing Effect Node

import torch
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextCrossProcessing:
    """
    APNext Cross Processing Effect Node
    Simulates cross processing film techniques with channel manipulation and curves
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "process_type": ([
                    "E6_in_C41", "C41_in_E6", "Custom", 
                    "Vintage_Warm", "Cold_Blue", "Green_Magenta"
                ], {"default": "E6_in_C41"}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.2, "min": 0.5, "max": 3.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.3, "min": 0.0, "max": 3.0, "step": 0.01}),
                # Custom channel mixing
                "red_in_red": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green_in_red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue_in_red": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "red_in_green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green_in_green": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue_in_green": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "red_in_blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "green_in_blue": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.01}),
                "blue_in_blue": ("FLOAT", {"default": 1.0, "min": -2.0, "max": 2.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_cross_processing"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_cross_processing(self, images, process_type, intensity, contrast, saturation,
                             red_in_red, green_in_red, blue_in_red,
                             red_in_green, green_in_green, blue_in_green,
                             red_in_blue, green_in_blue, blue_in_blue):
        """Apply cross processing effect to images"""
        device = images.device
        dtype = images.dtype
        
        # Get the appropriate color matrix
        if process_type == "Custom":
            color_matrix = torch.tensor([
                [red_in_red, green_in_red, blue_in_red],
                [red_in_green, green_in_green, blue_in_green],
                [red_in_blue, green_in_blue, blue_in_blue]
            ], device=device, dtype=dtype)
        else:
            color_matrix = self._get_preset_matrix(process_type, device, dtype)
        
        # Apply cross processing
        result = self._apply_cross_process_tensor(
            images, color_matrix, intensity, contrast, saturation
        )
        
        return (result,)

    def _get_preset_matrix(self, process_type, device, dtype):
        """Get predefined color matrices for different cross processing types"""
        matrices = {
            "E6_in_C41": torch.tensor([  # Slide film in negative chemistry
                [1.2, -0.1, 0.1],
                [0.0, 1.1, -0.1],
                [-0.1, 0.2, 1.3]
            ], device=device, dtype=dtype),
            
            "C41_in_E6": torch.tensor([  # Negative film in slide chemistry
                [0.9, 0.1, -0.1],
                [0.1, 1.0, 0.1],
                [0.2, -0.2, 0.8]
            ], device=device, dtype=dtype),
            
            "Vintage_Warm": torch.tensor([
                [1.3, 0.1, -0.1],
                [-0.1, 1.1, 0.2],
                [-0.2, -0.1, 0.9]
            ], device=device, dtype=dtype),
            
            "Cold_Blue": torch.tensor([
                [0.8, -0.1, 0.2],
                [0.1, 1.0, -0.1],
                [0.3, 0.2, 1.4]
            ], device=device, dtype=dtype),
            
            "Green_Magenta": torch.tensor([
                [1.1, -0.2, 0.1],
                [-0.3, 1.4, -0.1],
                [0.1, -0.2, 1.0]
            ], device=device, dtype=dtype),
        }
        
        return matrices.get(process_type, torch.eye(3, device=device, dtype=dtype))

    def _apply_cross_process_tensor(self, images, color_matrix, intensity, contrast, saturation):
        """Apply cross processing using tensor operations"""
        batch_size, height, width, channels = images.shape
        
        # Apply color matrix transformation
        images_flat = images.view(-1, 3)
        processed_flat = torch.matmul(images_flat, color_matrix.t())
        processed = processed_flat.view(batch_size, height, width, channels)
        
        # Apply contrast adjustment
        if contrast != 1.0:
            mid_point = 0.5
            processed = (processed - mid_point) * contrast + mid_point
        
        # Apply saturation adjustment
        if saturation != 1.0:
            processed = self._apply_saturation_tensor(processed, saturation)
        
        # Apply cross processing curves (S-curves for each channel)
        processed = self._apply_film_curves_tensor(processed)
        
        # Blend with original based on intensity
        result = images * (1 - intensity) + processed * intensity
        
        # Clamp values
        result = torch.clamp(result, 0, 1)
        return result

    def _apply_saturation_tensor(self, images, saturation):
        """Apply saturation adjustment using tensor operations"""
        # Calculate luminance
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        luminance = luminance.unsqueeze(-1)
        
        # Blend between grayscale and original based on saturation
        result = luminance + (images - luminance) * saturation
        return result

    def _apply_film_curves_tensor(self, images):
        """Apply characteristic film curves that create the cross-processing look"""
        # Create S-curve for each channel with different characteristics
        # Red channel: lifted shadows, compressed highlights
        red = images[:, :, :, 0:1]
        red_curved = self._s_curve_tensor(red, shadows=0.1, highlights=-0.1, contrast=1.1)
        
        # Green channel: more aggressive S-curve
        green = images[:, :, :, 1:2]
        green_curved = self._s_curve_tensor(green, shadows=0.05, highlights=-0.05, contrast=1.2)
        
        # Blue channel: inverted curve characteristics
        blue = images[:, :, :, 2:3]
        blue_curved = self._s_curve_tensor(blue, shadows=-0.05, highlights=0.15, contrast=1.1)
        
        return torch.cat([red_curved, green_curved, blue_curved], dim=3)

    def _s_curve_tensor(self, channel, shadows=0.0, highlights=0.0, contrast=1.0):
        """Apply S-curve to a single channel"""
        # Apply shadow/highlight adjustments
        adjusted = channel + shadows * (1 - channel) + highlights * channel
        
        # Apply contrast with S-curve
        # Use a sigmoid-based S-curve
        mid_point = 0.5
        normalized = (adjusted - mid_point) * contrast
        s_curved = torch.sigmoid(normalized * 2) 
        
        # Scale back to 0-1 range
        result = s_curved * (1 - mid_point) + mid_point
        
        return result
