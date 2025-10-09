# Image utility functions

import torch
import numpy as np
from PIL import Image

def tensor2pil(t_image: torch.Tensor) -> Image:
    """Convert ComfyUI tensor to PIL Image"""
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image: Image) -> torch.Tensor:
    """Convert PIL Image to ComfyUI tensor"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)
