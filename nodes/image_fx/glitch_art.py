# APNext Glitch Art Effect Node

import torch
import numpy as np
import random
from ...utils.constants import CUSTOM_CATEGORY


class APNextGlitchArt:
    """
    APNext Glitch Art Effect Node
    Creates digital corruption and glitch effects
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "glitch_type": ([
                    "RGB_Shift", "Data_Moshing", "Pixel_Sort", "Scanlines", 
                    "Digital_Noise", "Color_Channel_Shift", "Compression_Artifacts"
                ], {"default": "RGB_Shift"}),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "randomness": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_glitch_art"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_glitch_art(self, images, glitch_type, intensity, randomness, seed):
        """Apply glitch art effects to images"""
        # Set random seed
        random.seed(seed)
        torch.manual_seed(seed % (2**32))
        
        device = images.device
        dtype = images.dtype
        
        if glitch_type == "RGB_Shift":
            result = self._rgb_channel_shift(images, intensity, randomness)
        elif glitch_type == "Data_Moshing":
            result = self._data_moshing_effect(images, intensity, randomness)
        elif glitch_type == "Pixel_Sort":
            result = self._pixel_sorting_effect(images, intensity, randomness)
        elif glitch_type == "Scanlines":
            result = self._scanlines_effect(images, intensity, randomness)
        elif glitch_type == "Digital_Noise":
            result = self._digital_noise_effect(images, intensity, randomness)
        elif glitch_type == "Color_Channel_Shift":
            result = self._color_channel_shift(images, intensity, randomness)
        elif glitch_type == "Compression_Artifacts":
            result = self._compression_artifacts(images, intensity, randomness)
        else:
            result = images
        
        return (result,)

    def _rgb_channel_shift(self, images, intensity, randomness):
        """Create RGB channel shift glitch effect"""
        batch_size, height, width, channels = images.shape
        device = images.device
        
        # Calculate shift amounts
        max_shift = int(intensity * 20 + randomness * 10)
        
        # Create random shifts for each channel
        r_shift_x = random.randint(-max_shift, max_shift)
        r_shift_y = random.randint(-max_shift, max_shift)
        g_shift_x = random.randint(-max_shift, max_shift)
        g_shift_y = random.randint(-max_shift, max_shift)
        b_shift_x = random.randint(-max_shift, max_shift)
        b_shift_y = random.randint(-max_shift, max_shift)
        
        result = images.clone()
        
        # Apply shifts to each channel
        for batch_idx in range(batch_size):
            # Red channel shift
            if abs(r_shift_x) > 0 or abs(r_shift_y) > 0:
                shifted_red = torch.roll(images[batch_idx, :, :, 0], shifts=(r_shift_y, r_shift_x), dims=(0, 1))
                result[batch_idx, :, :, 0] = shifted_red
            
            # Green channel shift
            if abs(g_shift_x) > 0 or abs(g_shift_y) > 0:
                shifted_green = torch.roll(images[batch_idx, :, :, 1], shifts=(g_shift_y, g_shift_x), dims=(0, 1))
                result[batch_idx, :, :, 1] = shifted_green
            
            # Blue channel shift
            if abs(b_shift_x) > 0 or abs(b_shift_y) > 0:
                shifted_blue = torch.roll(images[batch_idx, :, :, 2], shifts=(b_shift_y, b_shift_x), dims=(0, 1))
                result[batch_idx, :, :, 2] = shifted_blue
        
        return torch.clamp(result, 0, 1)

    def _data_moshing_effect(self, images, intensity, randomness):
        """Create data moshing effect by corrupting image data"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Number of corruption blocks
        num_blocks = int(intensity * 50 + randomness * 20)
        
        for batch_idx in range(batch_size):
            for _ in range(num_blocks):
                # Random block position and size
                block_size = random.randint(5, int(intensity * 50 + 10))
                x = random.randint(0, width - block_size)
                y = random.randint(0, height - block_size)
                
                # Corruption type
                corruption_type = random.choice(['duplicate', 'scramble', 'invert', 'noise'])
                
                if corruption_type == 'duplicate':
                    # Duplicate nearby block
                    source_x = max(0, min(width - block_size, x + random.randint(-30, 30)))
                    source_y = max(0, min(height - block_size, y + random.randint(-30, 30)))
                    result[batch_idx, y:y+block_size, x:x+block_size] = \
                        images[batch_idx, source_y:source_y+block_size, source_x:source_x+block_size]
                
                elif corruption_type == 'scramble':
                    # Scramble pixels in block
                    block = result[batch_idx, y:y+block_size, x:x+block_size].clone()
                    flat_block = block.view(-1, channels)
                    indices = torch.randperm(flat_block.size(0))
                    scrambled = flat_block[indices].view(block_size, block_size, channels)
                    result[batch_idx, y:y+block_size, x:x+block_size] = scrambled
                
                elif corruption_type == 'invert':
                    # Invert colors in block
                    result[batch_idx, y:y+block_size, x:x+block_size] = \
                        1.0 - result[batch_idx, y:y+block_size, x:x+block_size]
                
                elif corruption_type == 'noise':
                    # Add noise to block
                    noise = torch.randn_like(result[batch_idx, y:y+block_size, x:x+block_size]) * 0.1
                    result[batch_idx, y:y+block_size, x:x+block_size] += noise
        
        return torch.clamp(result, 0, 1)

    def _pixel_sorting_effect(self, images, intensity, randomness):
        """Create pixel sorting glitch effect"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Number of sorting operations
        num_sorts = int(intensity * 20 + randomness * 10)
        
        for batch_idx in range(batch_size):
            for _ in range(num_sorts):
                # Random sorting direction and area
                if random.random() < 0.5:
                    # Horizontal sorting
                    row = random.randint(0, height - 1)
                    start_col = random.randint(0, width - 20)
                    end_col = min(width, start_col + random.randint(10, int(intensity * 100 + 20)))
                    
                    # Sort pixels by brightness
                    row_data = result[batch_idx, row, start_col:end_col]
                    brightness = 0.299 * row_data[:, 0] + 0.587 * row_data[:, 1] + 0.114 * row_data[:, 2]
                    sorted_indices = torch.argsort(brightness)
                    result[batch_idx, row, start_col:end_col] = row_data[sorted_indices]
                else:
                    # Vertical sorting
                    col = random.randint(0, width - 1)
                    start_row = random.randint(0, height - 20)
                    end_row = min(height, start_row + random.randint(10, int(intensity * 100 + 20)))
                    
                    # Sort pixels by brightness
                    col_data = result[batch_idx, start_row:end_row, col]
                    brightness = 0.299 * col_data[:, 0] + 0.587 * col_data[:, 1] + 0.114 * col_data[:, 2]
                    sorted_indices = torch.argsort(brightness)
                    result[batch_idx, start_row:end_row, col] = col_data[sorted_indices]
        
        return result

    def _scanlines_effect(self, images, intensity, randomness):
        """Create scanlines and CRT-like glitch effects"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Scanline parameters
        line_spacing = max(2, int(10 - intensity * 8))
        line_intensity = intensity * 0.5
        
        for batch_idx in range(batch_size):
            # Add horizontal scanlines
            for y in range(0, height, line_spacing):
                if random.random() < 0.7:  # Not every line
                    # Darken scanline
                    result[batch_idx, y] *= (1.0 - line_intensity)
                    
                    # Add some color shift
                    if random.random() < randomness:
                        shift = random.randint(-2, 2)
                        if shift != 0:
                            result[batch_idx, y] = torch.roll(result[batch_idx, y], shift, dims=0)
            
            # Add random glitch lines
            num_glitch_lines = int(intensity * 5 + randomness * 3)
            for _ in range(num_glitch_lines):
                y = random.randint(0, height - 1)
                # Duplicate or shift line
                if random.random() < 0.5:
                    # Duplicate from nearby line
                    source_y = max(0, min(height - 1, y + random.randint(-5, 5)))
                    result[batch_idx, y] = result[batch_idx, source_y]
                else:
                    # Color shift
                    result[batch_idx, y, :, 0] = torch.roll(result[batch_idx, y, :, 0], random.randint(-10, 10))
        
        return torch.clamp(result, 0, 1)

    def _digital_noise_effect(self, images, intensity, randomness):
        """Add digital noise and artifacts"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Add random digital noise
        noise_amount = intensity * randomness * 0.1
        noise = torch.randn_like(result) * noise_amount
        result += noise
        
        # Add salt and pepper noise
        salt_pepper_prob = intensity * randomness * 0.05
        mask = torch.rand_like(result[:, :, :, 0:1]) < salt_pepper_prob
        salt_pepper = torch.rand_like(result) > 0.5
        result = torch.where(mask, salt_pepper.float(), result)
        
        # Add compression-like artifacts
        if intensity > 0.3:
            # Quantize colors
            quantization_levels = max(8, int(256 - intensity * 200))
            result = torch.round(result * quantization_levels) / quantization_levels
        
        return torch.clamp(result, 0, 1)

    def _color_channel_shift(self, images, intensity, randomness):
        """Shift color channels in different directions"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Create different shift patterns for each channel
        for batch_idx in range(batch_size):
            # Red channel - horizontal shift
            h_shift = int(intensity * randomness * 20)
            if h_shift > 0:
                result[batch_idx, :, :, 0] = torch.roll(result[batch_idx, :, :, 0], h_shift, dims=1)
            
            # Green channel - vertical shift
            v_shift = int(intensity * randomness * 15)
            if v_shift > 0:
                result[batch_idx, :, :, 1] = torch.roll(result[batch_idx, :, :, 1], v_shift, dims=0)
            
            # Blue channel - diagonal effect
            if intensity > 0.5:
                # Create diagonal shift effect
                for y in range(height):
                    shift_amount = int((y / height) * intensity * 10)
                    if shift_amount > 0:
                        result[batch_idx, y, :, 2] = torch.roll(result[batch_idx, y, :, 2], shift_amount)
        
        return torch.clamp(result, 0, 1)

    def _compression_artifacts(self, images, intensity, randomness):
        """Simulate compression artifacts and blocking"""
        result = images.clone()
        batch_size, height, width, channels = images.shape
        
        # Block size for compression artifacts
        block_size = max(4, int(16 - intensity * 10))
        
        for batch_idx in range(batch_size):
            # Create blocking artifacts
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    end_y = min(height, y + block_size)
                    end_x = min(width, x + block_size)
                    
                    if random.random() < intensity * randomness:
                        # Average the block (compression-like effect)
                        block = result[batch_idx, y:end_y, x:end_x]
                        avg_color = torch.mean(block, dim=(0, 1), keepdim=True)
                        result[batch_idx, y:end_y, x:end_x] = avg_color
                    
                    elif random.random() < intensity * 0.3:
                        # Add ringing artifacts around edges
                        block = result[batch_idx, y:end_y, x:end_x]
                        # Simple edge detection
                        if block.size(0) > 1 and block.size(1) > 1:
                            # Calculate edges safely with proper size handling
                            vertical_edges = torch.abs(block[1:, :, :] - block[:-1, :, :])
                            horizontal_edges = torch.abs(block[:, 1:, :] - block[:, :-1, :])
                            
                            # Pad edges to match original block size
                            vertical_edges_padded = torch.zeros_like(block)
                            horizontal_edges_padded = torch.zeros_like(block)
                            
                            vertical_edges_padded[:-1, :, :] = vertical_edges
                            horizontal_edges_padded[:, :-1, :] = horizontal_edges
                            
                            edges = vertical_edges_padded + horizontal_edges_padded
                            edge_strength = torch.mean(edges)
                            
                            if edge_strength > 0.1:
                                # Add ringing
                                ringing = torch.sin(torch.arange(block.size(0), device=images.device).float() * 3.14159) * 0.05 * intensity
                                ringing = ringing.view(-1, 1, 1).expand_as(block)
                                result[batch_idx, y:end_y, x:end_x] += ringing
        
        return torch.clamp(result, 0, 1)
