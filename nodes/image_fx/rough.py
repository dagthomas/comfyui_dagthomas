# APNext Rough Effect Node

import torch
import numpy as np
from PIL import Image
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil, pil2tensor


class APNextRough:
    """
    APNext Rough Effect Node
    Creates a rough, posterized effect by reducing the image to a limited color palette
    Based on numpy implementation for better performance
    """
    
    def __init__(self):
        # Simple cache for color palettes
        self.palette_cache = {}
        self.max_cache_size = 10
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "color_count": ("INT", {"default": 24, "min": 4, "max": 64, "step": 1}),
                "aa_factor": ("INT", {"default": 4, "min": 1, "max": 8, "step": 1}),
                "resize_method": (["BICUBIC", "LANCZOS", "BILINEAR", "NEAREST"], {"default": "BICUBIC"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_rough"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX"

    def apply_rough(self, images, color_count, aa_factor, resize_method):
        """Apply rough/posterized effect to images"""
        result_images = []
        
        for image in images:
            # Convert tensor to PIL
            pil_image = tensor2pil(image)
            
            # Apply rough effect
            rough_image = self._create_rough_effect(
                pil_image, color_count, aa_factor, resize_method
            )
            
            # Convert back to tensor
            result_tensor = pil2tensor(rough_image)
            result_images.append(result_tensor)
        
        # Stack all processed images
        return (torch.cat(result_images, dim=0),)

    def _create_rough_effect(self, image, color_count, aa_factor, resize_method):
        """Create rough effect on a PIL image using numpy for efficiency"""
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert PIL to numpy array
        img = np.asarray(image, dtype='int64')
        height, width = img.shape[:2]
        
        # Try to get cached palette
        cache_key = self._get_palette_cache_key(img, color_count)
        if cache_key in self.palette_cache:
            colors = self.palette_cache[cache_key]
            print("Using cached palette.")
        else:
            # Generate new palette
            colors = self._generate_palette(img, color_count)
            
            # Cache the palette
            if len(self.palette_cache) >= self.max_cache_size:
                # Remove oldest entry
                oldest_key = next(iter(self.palette_cache))
                del self.palette_cache[oldest_key]
            self.palette_cache[cache_key] = colors
            
        print("Created palette.")
        print(colors)
        
        # Compute color distance of each pixel to each palette color (vectorized)
        colors_array = np.array(colors)
        img_flat = img.reshape(-1, 3)
        
        # Vectorized distance computation
        distances = np.abs(img_flat[:, None, :] - colors_array[None, :, :]).sum(axis=2)
        closest_indices = np.argmin(distances, axis=1)
        closest_indices = closest_indices.reshape(height, width)
        
        # Create new image with anti-aliasing
        new_img = np.zeros((height * aa_factor, width * aa_factor, 3), dtype=np.uint8)
        
        # Vectorized assignment of colors (much faster than nested loops)
        for y in range(height):
            for x in range(width):
                col = colors[closest_indices[y, x]]
                # Draw rectangles using numpy slicing
                new_img[aa_factor * y : aa_factor * y + aa_factor, 
                       aa_factor * x : aa_factor * x + aa_factor, :] = col
        
        print("Finished rough image.")
        
        # Convert back to PIL and resize
        nim = Image.fromarray(new_img)
        resize_filter = getattr(Image, resize_method)
        aim = nim.resize((width, height), resize_filter)
        
        return aim
    
    def _get_palette_cache_key(self, img, color_count):
        """Generate cache key for palette based on image statistics and color count"""
        # Use image statistics for cache key (much faster than hashing entire image)
        img_sample = img[::10, ::10]  # Sample every 10th pixel
        stats = (
            img_sample.mean(),
            img_sample.std(),
            img_sample.min(),
            img_sample.max(),
            color_count
        )
        return hash(stats)
    
    def _generate_palette(self, img, color_count):
        """Generate color palette using optimized algorithm"""
        height, width = img.shape[:2]
        
        # Initialize color palette with black and white
        colors = [(255, 255, 255), (0, 0, 0)]
        colors_array = np.array(colors)
        
        for i in range(color_count - 2):
            # Vectorized computation of distances to all existing colors
            img_expanded = img.reshape(-1, 3)  # Flatten image
            
            # Compute Manhattan distance to all existing colors at once
            distances = np.abs(img_expanded[:, None, :] - colors_array[None, :, :]).sum(axis=2)
            min_distances = distances.min(axis=1)
            
            # Find pixel with maximum minimum distance (furthest from all existing colors)
            max_idx = np.argmax(min_distances)
            new_color = img_expanded[max_idx]
            
            print(f"Added color: {new_color}")
            colors.append(tuple(new_color))
            colors_array = np.array(colors)
            
        return colors
