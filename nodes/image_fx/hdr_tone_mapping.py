# APNext HDR Tone Mapping Effect Node

import torch
import numpy as np
from ...utils.constants import CUSTOM_CATEGORY


class APNextHDRToneMapping:
    """
    APNext HDR Tone Mapping Effect Node
    Simulates HDR tone mapping effects for dramatic and surreal looks
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "method": ([
                    "Reinhard", "Drago", "Mantiuk", "Photographic", 
                    "Adaptive_Log", "Filmic", "ACES"
                ], {"default": "Reinhard"}),
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 2.2, "min": 0.5, "max": 4.0, "step": 0.01}),
                "intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                # Method-specific parameters
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.01}),
                "key_value": ("FLOAT", {"default": 0.18, "min": 0.01, "max": 1.0, "step": 0.01}),
                "adaptation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_correction": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                "local_adaptation": ("BOOLEAN", {"default": False}),
                "radius": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_hdr_tone_mapping"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_hdr_tone_mapping(self, images, method, exposure, gamma, intensity,
                              white_point, key_value, adaptation, color_correction,
                              local_adaptation, radius):
        """Apply HDR tone mapping effect to images"""
        device = images.device
        dtype = images.dtype
        
        # Apply exposure adjustment first
        if exposure != 0.0:
            exposure_factor = 2.0 ** exposure
            images = images * exposure_factor
        
        # Apply tone mapping based on selected method
        if method == "Reinhard":
            result = self._reinhard_tone_mapping(images, white_point, local_adaptation, radius)
        elif method == "Drago":
            result = self._drago_tone_mapping(images, adaptation, white_point)
        elif method == "Mantiuk":
            result = self._mantiuk_tone_mapping(images, color_correction, adaptation)
        elif method == "Photographic":
            result = self._photographic_tone_mapping(images, key_value, white_point)
        elif method == "Adaptive_Log":
            result = self._adaptive_log_tone_mapping(images, adaptation)
        elif method == "Filmic":
            result = self._filmic_tone_mapping(images, white_point)
        elif method == "ACES":
            result = self._aces_tone_mapping(images)
        else:
            result = images
        
        # Apply gamma correction
        if gamma != 1.0:
            result = torch.pow(torch.clamp(result, 0.001, 1.0), 1.0 / gamma)
        
        # Blend with original based on intensity
        result = images * (1 - intensity) + result * intensity
        
        # Clamp final result
        result = torch.clamp(result, 0, 1)
        return (result,)

    def _reinhard_tone_mapping(self, images, white_point, local_adaptation, radius):
        """Reinhard tone mapping operator"""
        # Calculate luminance
        luminance = self._calculate_luminance(images)
        
        if local_adaptation:
            # Local adaptation version (simplified)
            local_lum = self._gaussian_blur_luminance(luminance, radius)
            adaptation_lum = local_lum / (local_lum + 1.0)
            tone_mapped_lum = luminance / (luminance + adaptation_lum)
        else:
            # Global version
            tone_mapped_lum = luminance / (luminance + white_point ** 2)
        
        # Apply to color channels
        result = self._apply_luminance_change(images, luminance, tone_mapped_lum)
        return result

    def _drago_tone_mapping(self, images, adaptation, white_point):
        """Drago tone mapping operator"""
        luminance = self._calculate_luminance(images)
        
        # Drago's adaptive logarithmic mapping
        log_lum = torch.log10(torch.clamp(luminance, 1e-6, float('inf')))
        log_white = torch.log10(white_point)
        
        # Bias function
        bias = torch.pow(adaptation, log_lum / log_white)
        
        # Apply tone mapping
        tone_mapped_lum = (log_lum / log_white) / torch.log10(2 + 8 * bias)
        tone_mapped_lum = torch.clamp(tone_mapped_lum, 0, 1)
        
        result = self._apply_luminance_change(images, luminance, tone_mapped_lum)
        return result

    def _mantiuk_tone_mapping(self, images, color_correction, adaptation):
        """Mantiuk tone mapping operator (simplified version)"""
        luminance = self._calculate_luminance(images)
        
        # Logarithmic compression
        log_lum = torch.log(torch.clamp(luminance, 1e-6, float('inf')))
        
        # Adaptive factor
        adaptive_factor = adaptation * torch.tanh(log_lum)
        
        # Tone mapping
        tone_mapped_lum = torch.exp(log_lum * adaptive_factor)
        tone_mapped_lum = tone_mapped_lum / (tone_mapped_lum + 1.0)
        
        # Color correction
        if color_correction != 1.0:
            saturation_factor = 1.0 + (color_correction - 1.0) * (1.0 - tone_mapped_lum)
            result = self._apply_luminance_change(images, luminance, tone_mapped_lum)
            result = self._apply_saturation_tensor(result, saturation_factor.mean().item())
        else:
            result = self._apply_luminance_change(images, luminance, tone_mapped_lum)
        
        return result

    def _photographic_tone_mapping(self, images, key_value, white_point):
        """Photographic tone reproduction operator"""
        luminance = self._calculate_luminance(images)
        
        # Calculate log-average luminance
        log_avg = torch.exp(torch.mean(torch.log(torch.clamp(luminance, 1e-6, float('inf')))))
        
        # Scale luminance
        scaled_lum = (key_value / log_avg) * luminance
        
        # Apply tone mapping with white point
        tone_mapped_lum = scaled_lum * (1.0 + scaled_lum / (white_point ** 2)) / (1.0 + scaled_lum)
        
        result = self._apply_luminance_change(images, luminance, tone_mapped_lum)
        return result

    def _adaptive_log_tone_mapping(self, images, adaptation):
        """Adaptive logarithmic tone mapping"""
        luminance = self._calculate_luminance(images)
        
        # Logarithmic mapping with adaptation
        log_lum = torch.log(torch.clamp(luminance, 1e-6, float('inf')))
        max_log = torch.max(log_lum)
        min_log = torch.min(log_lum)
        
        # Normalize and apply adaptation
        normalized = (log_lum - min_log) / (max_log - min_log + 1e-6)
        adapted = torch.pow(normalized, adaptation)
        
        result = self._apply_luminance_change(images, luminance, adapted)
        return result

    def _filmic_tone_mapping(self, images, white_point):
        """Filmic tone mapping (Uncharted 2 style)"""
        def filmic_curve(x):
            A, B, C, D, E, F = 0.15, 0.50, 0.10, 0.20, 0.02, 0.30
            return ((x * (A * x + C * B) + D * E) / (x * (A * x + B) + D * F)) - E / F
        
        # Apply filmic curve
        tone_mapped = filmic_curve(images * 2.0) / filmic_curve(torch.tensor(white_point))
        
        return torch.clamp(tone_mapped, 0, 1)

    def _aces_tone_mapping(self, images):
        """ACES filmic tone mapping"""
        # ACES RRT/ODT approximation
        a, b, c, d, e = 2.51, 0.03, 2.43, 0.59, 0.14
        
        tone_mapped = (images * (a * images + b)) / (images * (c * images + d) + e)
        
        return torch.clamp(tone_mapped, 0, 1)

    def _calculate_luminance(self, images):
        """Calculate luminance from RGB"""
        return 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]

    def _apply_luminance_change(self, images, old_luminance, new_luminance):
        """Apply luminance change while preserving color ratios"""
        old_luminance = old_luminance.unsqueeze(-1)
        new_luminance = new_luminance.unsqueeze(-1)
        
        # Avoid division by zero
        ratio = torch.where(old_luminance > 1e-6, new_luminance / old_luminance, torch.ones_like(old_luminance))
        
        result = images * ratio
        return torch.clamp(result, 0, 1)

    def _gaussian_blur_luminance(self, luminance, radius):
        """Apply gaussian blur to luminance for local adaptation"""
        # Simple box blur approximation for performance
        kernel_size = max(3, int(radius * 20))
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create simple averaging kernel
        kernel = torch.ones(1, 1, kernel_size, kernel_size, device=luminance.device) / (kernel_size ** 2)
        
        # Add batch and channel dimensions for conv2d
        lum_for_conv = luminance.unsqueeze(1)  # [batch, 1, height, width]
        
        # Apply convolution
        padding = kernel_size // 2
        blurred = torch.nn.functional.conv2d(lum_for_conv, kernel, padding=padding)
        
        # Remove extra dimension
        return blurred.squeeze(1)

    def _apply_saturation_tensor(self, images, saturation):
        """Apply saturation adjustment using tensor operations"""
        # Calculate luminance
        luminance = self._calculate_luminance(images)
        luminance = luminance.unsqueeze(-1)
        
        # Blend between grayscale and original based on saturation
        result = luminance + (images - luminance) * saturation
        return torch.clamp(result, 0, 1)
