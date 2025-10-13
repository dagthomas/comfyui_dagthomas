# APNext Noise Effect Node

import torch
from PIL import Image, ImageDraw
import numpy as np
import random
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil, pil2tensor

try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class APNextNoise:
    """
    APNext Noise Effect Node
    Adds noise to images - can be B&W with transparency or colored based on dominant image colors
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01}),
                "noise_type": (["monochrome", "colored", "film_grain"], {"default": "monochrome"}),
                "blend_mode": (["overlay", "multiply", "screen", "soft_light"], {"default": "overlay"}),
                "grain_size": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "color_count": ("INT", {"default": 3, "min": 2, "max": 8, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_noise"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX"

    def apply_noise(self, images, intensity, noise_type, blend_mode, grain_size, color_count, seed):
        """Apply noise effect to images"""
        result_images = []
        
        # Set random seed for reproducible results
        random.seed(seed)
        # For numpy, we need to clamp to 32-bit range
        np.random.seed(seed % (2**32))
        
        for image in images:
            # Convert tensor to PIL
            pil_image = tensor2pil(image)
            
            # Apply noise effect
            noisy_image = self._create_noise_effect(
                pil_image, intensity, noise_type, blend_mode, grain_size, color_count
            )
            
            # Convert back to tensor
            result_tensor = pil2tensor(noisy_image)
            result_images.append(result_tensor)
        
        # Stack all processed images
        return (torch.cat(result_images, dim=0),)

    def _create_noise_effect(self, image, intensity, noise_type, blend_mode, grain_size, color_count):
        """Create noise effect on a PIL image"""
        # Convert to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        width, height = image.size
        
        if noise_type == "monochrome":
            noise_layer = self._create_monochrome_noise(width, height, grain_size)
        elif noise_type == "colored":
            dominant_colors = self._extract_dominant_colors(image, color_count)
            noise_layer = self._create_colored_noise(width, height, dominant_colors, grain_size)
        elif noise_type == "film_grain":
            noise_layer = self._create_film_grain(width, height, grain_size)
        
        # Blend the noise with the original image
        result = self._blend_noise(image, noise_layer, blend_mode, intensity)
        
        return result

    def _create_monochrome_noise(self, width, height, grain_size):
        """Create black and white noise with transparency"""
        # Create noise array
        noise_size = int(max(width, height) / grain_size)
        noise = np.random.randint(0, 256, (noise_size, noise_size), dtype=np.uint8)
        
        # Create PIL image from noise
        noise_img = Image.fromarray(noise, mode='L')
        
        # Resize to match target dimensions
        noise_img = noise_img.resize((width, height), Image.NEAREST if grain_size > 1 else Image.LANCZOS)
        
        # Convert to RGBA with transparency based on noise intensity
        noise_rgba = Image.new('RGBA', (width, height))
        noise_array = np.array(noise_img)
        
        # Create RGBA array where alpha is based on the noise value
        rgba_array = np.zeros((height, width, 4), dtype=np.uint8)
        rgba_array[:, :, 0] = noise_array  # R
        rgba_array[:, :, 1] = noise_array  # G
        rgba_array[:, :, 2] = noise_array  # B
        rgba_array[:, :, 3] = noise_array  # A (transparency based on noise)
        
        return Image.fromarray(rgba_array, 'RGBA')

    def _create_colored_noise(self, width, height, colors, grain_size):
        """Create colored noise using dominant colors from the image"""
        # Create noise array
        noise_size = int(max(width, height) / grain_size)
        
        # Create RGB noise using the dominant colors
        noise_r = np.random.randint(0, 256, (noise_size, noise_size), dtype=np.uint8)
        noise_g = np.random.randint(0, 256, (noise_size, noise_size), dtype=np.uint8)
        noise_b = np.random.randint(0, 256, (noise_size, noise_size), dtype=np.uint8)
        
        # Map noise values to dominant colors
        color_indices = np.random.choice(len(colors), (noise_size, noise_size))
        
        # Create colored noise
        colored_noise = np.zeros((noise_size, noise_size, 3), dtype=np.uint8)
        for i, color in enumerate(colors):
            mask = color_indices == i
            colored_noise[mask] = color
        
        # Add some randomness to the colors
        variation = np.random.randint(-30, 31, (noise_size, noise_size, 3), dtype=np.int16)
        colored_noise = np.clip(colored_noise.astype(np.int16) + variation, 0, 255).astype(np.uint8)
        
        # Create PIL image
        noise_img = Image.fromarray(colored_noise)
        
        # Resize to match target dimensions
        noise_img = noise_img.resize((width, height), Image.NEAREST if grain_size > 1 else Image.LANCZOS)
        
        return noise_img

    def _create_film_grain(self, width, height, grain_size):
        """Create film grain effect"""
        # Create multiple layers of noise for more realistic grain
        noise_size = int(max(width, height) / grain_size)
        
        # Fine grain
        fine_grain = np.random.normal(0, 0.3, (noise_size, noise_size))
        # Coarse grain
        coarse_grain = np.random.normal(0, 0.1, (noise_size // 2, noise_size // 2))
        coarse_grain = np.repeat(np.repeat(coarse_grain, 2, axis=0), 2, axis=1)[:noise_size, :noise_size]
        
        # Combine grains
        combined_grain = fine_grain + coarse_grain
        
        # Convert to 0-255 range
        grain_normalized = ((combined_grain + 1) * 127.5).clip(0, 255).astype(np.uint8)
        
        # Create RGB grain
        grain_rgb = np.stack([grain_normalized] * 3, axis=2)
        
        # Create PIL image
        grain_img = Image.fromarray(grain_rgb)
        
        # Resize to match target dimensions
        grain_img = grain_img.resize((width, height), Image.NEAREST if grain_size > 1 else Image.LANCZOS)
        
        return grain_img

    def _extract_dominant_colors(self, image, n_colors):
        """Extract dominant colors from the image using optimized method"""
        # Convert image to numpy array and reshape for clustering
        img_array = np.array(image)
        pixels = img_array.reshape(-1, 3)
        
        # Sample pixels for performance (use every 20th pixel for speed)
        sampled_pixels = pixels[::20]
        
        if SKLEARN_AVAILABLE:
            # Use K-means clustering if available
            try:
                kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                kmeans.fit(sampled_pixels)
                colors = kmeans.cluster_centers_.astype(int)
                return colors.tolist()
            except:
                pass
        
        # Fallback: use histogram-based color extraction (faster than K-means)
        return self._extract_colors_histogram(sampled_pixels, n_colors)
    
    def _extract_colors_histogram(self, pixels, n_colors):
        """Extract colors using histogram-based method (faster fallback)"""
        # Quantize colors to reduce search space
        quantized = (pixels // 32) * 32  # Reduce to ~8 levels per channel
        
        # Find unique colors and their counts
        unique_colors, counts = np.unique(quantized.reshape(-1, quantized.shape[-1]), 
                                         axis=0, return_counts=True)
        
        # Get the most frequent colors
        most_frequent_indices = np.argsort(counts)[-n_colors:]
        dominant_colors = unique_colors[most_frequent_indices]
        
        # Add some variation to avoid too uniform colors
        variation = np.random.randint(-16, 17, dominant_colors.shape)
        dominant_colors = np.clip(dominant_colors + variation, 0, 255)
        
        return dominant_colors.tolist()

    def _blend_noise(self, base_image, noise_layer, blend_mode, intensity):
        """Blend noise with the base image using specified blend mode"""
        # Convert to numpy arrays
        base_array = np.array(base_image, dtype=np.float32) / 255.0
        
        # Handle noise layer (might be RGBA or RGB)
        if noise_layer.mode == 'RGBA':
            noise_array = np.array(noise_layer, dtype=np.float32)
            noise_rgb = noise_array[:, :, :3] / 255.0
            alpha = noise_array[:, :, 3] / 255.0
        else:
            noise_rgb = np.array(noise_layer, dtype=np.float32) / 255.0
            alpha = np.ones((noise_rgb.shape[0], noise_rgb.shape[1])) * 0.5
        
        # Apply intensity to alpha
        alpha = alpha * intensity
        
        # Apply blend mode
        if blend_mode == "overlay":
            # Overlay blend mode
            mask = base_array < 0.5
            blended = np.where(
                mask,
                2 * base_array * noise_rgb,
                1 - 2 * (1 - base_array) * (1 - noise_rgb)
            )
        elif blend_mode == "multiply":
            blended = base_array * noise_rgb
        elif blend_mode == "screen":
            blended = 1 - (1 - base_array) * (1 - noise_rgb)
        elif blend_mode == "soft_light":
            blended = np.where(
                noise_rgb <= 0.5,
                base_array - (1 - 2 * noise_rgb) * base_array * (1 - base_array),
                base_array + (2 * noise_rgb - 1) * (np.sqrt(base_array) - base_array)
            )
        
        # Apply alpha blending
        alpha_expanded = np.expand_dims(alpha, axis=2)
        result = base_array * (1 - alpha_expanded) + blended * alpha_expanded
        
        # Clamp and convert back to 0-255
        result = np.clip(result, 0, 1) * 255
        
        return Image.fromarray(result.astype(np.uint8))
