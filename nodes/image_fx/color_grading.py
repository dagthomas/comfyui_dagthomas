# APNext Color Grading Effect Node

import torch
import numpy as np
from PIL import Image
import os
import glob
import folder_paths
from ...utils.constants import CUSTOM_CATEGORY
from ...utils.image_utils import tensor2pil, pil2tensor


class APNextColorGrading:
    """
    APNext Color Grading Effect Node
    Applies color grading using LUT files or manual controls
    
    Supported LUT formats:
    - .cube files (Adobe/Blackmagic)
    - .3dl files (Autodesk/Flame)  
    - Image LUTs (.png, .jpg, .tiff, .exr)
    
    Example paths:
    - C:/LUTs/cinematic.cube
    - ./assets/vintage_film.3dl
    - /path/to/lut_image.png
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        # Register LUT file types with ComfyUI's folder_paths system
        cls._register_lut_paths()
        
        # Get LUT files using ComfyUI's standard system
        lut_files = folder_paths.get_filename_list("luts")
        if not lut_files:
            lut_files = ["None"]
        
        return {
            "required": {
                "images": ("IMAGE",),
                "method": (["manual", "lut_file"], {"default": "manual"}),
                "lut_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01}),
                # Manual controls
                "exposure": ("FLOAT", {"default": 0.0, "min": -3.0, "max": 3.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "highlights": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "shadows": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.01}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "lut_file": (lut_files, {"default": "None"}),
                "custom_lut_path": ("STRING", {
                    "default": "", 
                    "multiline": False,
                    "placeholder": "Custom LUT path (overrides picker)"
                }),
            }
        }
    
    @classmethod
    def _register_lut_paths(cls):
        """Register LUT file paths with ComfyUI's folder system"""
        try:
            # Add LUT directory to ComfyUI's folder paths
            if "luts" not in folder_paths.folder_names_and_paths:
                lut_dirs = [
                    os.path.join(folder_paths.models_dir, "luts"),
                    os.path.join(folder_paths.input_directory, "luts"),
                    "./luts",
                    "./LUTs"
                ]
                
                # Create luts directory in models if it doesn't exist
                main_lut_dir = os.path.join(folder_paths.models_dir, "luts")
                if not os.path.exists(main_lut_dir):
                    os.makedirs(main_lut_dir, exist_ok=True)
                    print(f"📁 Created LUT directory: {main_lut_dir}")
                
                # Register with ComfyUI
                folder_paths.folder_names_and_paths["luts"] = (lut_dirs, {".cube", ".3dl", ".png", ".jpg", ".jpeg", ".tiff", ".exr"})
                print("✅ Registered LUT file paths with ComfyUI")
        except Exception as e:
            print(f"⚠️ Could not register LUT paths: {e}")
            # Fallback to manual scanning
            pass
    
    @classmethod
    def _scan_for_lut_files(cls):
        """Scan common directories for LUT files"""
        lut_files = ["None"]  # Default option
        
        # Common LUT directories to scan
        search_dirs = [
            "./luts",
            "./LUTs", 
            "./assets/luts",
            "./custom_nodes/comfyui_dagthomas/luts",
            os.path.expanduser("~/LUTs"),
            os.path.expanduser("~/Documents/LUTs"),
            "C:/LUTs" if os.name == 'nt' else "/usr/local/share/luts",
        ]
        
        # LUT file extensions
        extensions = ['*.cube', '*.3dl', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.exr']
        
        for directory in search_dirs:
            if os.path.exists(directory):
                for ext in extensions:
                    pattern = os.path.join(directory, '**', ext)
                    files = glob.glob(pattern, recursive=True)
                    for file_path in files:
                        # Create a display name (relative path or just filename)
                        if len(file_path) > 60:
                            display_name = f"...{file_path[-57:]}"
                        else:
                            display_name = file_path
                        lut_files.append(f"{display_name}|{file_path}")
        
        # Also scan current working directory
        for ext in extensions:
            files = glob.glob(ext)
            for file_path in files:
                lut_files.append(f"{file_path}|{os.path.abspath(file_path)}")
        
        return lut_files

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply_color_grading"
    CATEGORY = f"{CUSTOM_CATEGORY}/APNext FX Advanced"

    def apply_color_grading(self, images, method, lut_strength, exposure, contrast, 
                          highlights, shadows, saturation, temperature, tint, 
                          lut_file="None", custom_lut_path=""):
        """Apply color grading to images"""
        device = images.device
        dtype = images.dtype
        
        if method == "lut_file":
            # Determine which LUT path to use
            lut_path = None
            
            # Custom path takes priority
            if custom_lut_path and custom_lut_path.strip():
                lut_path = custom_lut_path.strip()
                print(f"📁 Using custom LUT path: {lut_path}")
            
            # Otherwise use ComfyUI file selection
            elif lut_file and lut_file != "None":
                lut_path = folder_paths.get_full_path("luts", lut_file)
                print(f"📁 Using selected LUT: {lut_path}")
            
            # Apply LUT if we have a valid path
            if lut_path and os.path.exists(lut_path):
                result = self._apply_lut_grading(images, lut_path, lut_strength)
            elif lut_path:
                print(f"⚠️ LUT file not found: {lut_path}")
                print("Using manual grading instead.")
                result = self._apply_manual_grading(
                    images, exposure, contrast, highlights, shadows, 
                    saturation, temperature, tint
                )
            else:
                print("⚠️ No LUT file selected. Using manual grading instead.")
                result = self._apply_manual_grading(
                    images, exposure, contrast, highlights, shadows, 
                    saturation, temperature, tint
                )
        else:
            result = self._apply_manual_grading(
                images, exposure, contrast, highlights, shadows, 
                saturation, temperature, tint
            )
        
        return (result,)

    def _apply_lut_grading(self, images, lut_path, strength):
        """Apply LUT-based color grading"""
        try:
            # Load LUT file
            lut = self._load_lut_file(lut_path)
            if lut is None:
                return images
            
            # Apply LUT to images
            result = self._apply_lut_tensor(images, lut, strength)
            return result
            
        except Exception as e:
            print(f"Error applying LUT: {e}")
            return images

    def _apply_manual_grading(self, images, exposure, contrast, highlights, shadows, 
                            saturation, temperature, tint):
        """Apply manual color grading controls"""
        result = images.clone()
        
        # Apply exposure
        if exposure != 0.0:
            exposure_factor = 2.0 ** exposure
            result = result * exposure_factor
        
        # Apply contrast
        if contrast != 1.0:
            # Contrast around middle gray (0.18 in linear, ~0.5 in sRGB)
            mid_gray = 0.5
            result = (result - mid_gray) * contrast + mid_gray
        
        # Apply highlights and shadows
        if highlights != 0.0 or shadows != 0.0:
            result = self._apply_highlight_shadow_tensor(result, highlights, shadows)
        
        # Apply saturation
        if saturation != 1.0:
            result = self._apply_saturation_tensor(result, saturation)
        
        # Apply temperature and tint
        if temperature != 0.0 or tint != 0.0:
            result = self._apply_white_balance_tensor(result, temperature, tint)
        
        # Clamp values
        result = torch.clamp(result, 0, 1)
        return result

    def _apply_highlight_shadow_tensor(self, images, highlights, shadows):
        """Apply highlight and shadow adjustments using tensor operations"""
        # Calculate luminance
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        luminance = luminance.unsqueeze(-1)
        
        # Create masks for highlights and shadows
        highlight_mask = torch.sigmoid((luminance - 0.7) * 10)  # Soft transition around 0.7
        shadow_mask = torch.sigmoid((0.3 - luminance) * 10)    # Soft transition around 0.3
        
        # Apply adjustments
        highlight_adj = 1.0 + highlights * highlight_mask
        shadow_adj = 1.0 + shadows * shadow_mask
        
        result = images * highlight_adj * shadow_adj
        return result

    def _apply_saturation_tensor(self, images, saturation):
        """Apply saturation adjustment using tensor operations"""
        # Calculate luminance
        luminance = 0.299 * images[:, :, :, 0] + 0.587 * images[:, :, :, 1] + 0.114 * images[:, :, :, 2]
        luminance = luminance.unsqueeze(-1)
        
        # Blend between grayscale and original based on saturation
        result = luminance + (images - luminance) * saturation
        return result

    def _apply_white_balance_tensor(self, images, temperature, tint):
        """Apply white balance (temperature and tint) using tensor operations"""
        # Temperature affects blue-yellow balance
        if temperature != 0.0:
            # Warm up (positive) or cool down (negative)
            temp_matrix = torch.tensor([
                [1.0 + temperature * 0.2, 0.0, -temperature * 0.1],
                [0.0, 1.0, 0.0],
                [-temperature * 0.1, 0.0, 1.0 - temperature * 0.2]
            ], device=images.device, dtype=images.dtype)
            
            # Apply color matrix
            images_flat = images.view(-1, 3)
            images_adjusted = torch.matmul(images_flat, temp_matrix.t())
            images = images_adjusted.view(images.shape)
        
        # Tint affects green-magenta balance
        if tint != 0.0:
            tint_matrix = torch.tensor([
                [1.0 - tint * 0.1, tint * 0.2, 0.0],
                [-tint * 0.1, 1.0 + tint * 0.1, 0.0],
                [0.0, 0.0, 1.0]
            ], device=images.device, dtype=images.dtype)
            
            # Apply color matrix
            images_flat = images.view(-1, 3)
            images_adjusted = torch.matmul(images_flat, tint_matrix.t())
            images = images_adjusted.view(images.shape)
        
        return images

    def _load_lut_file(self, lut_path):
        """Load LUT file (supports .cube, .3dl, and image formats)"""
        try:
            file_ext = lut_path.lower()
            if file_ext.endswith('.cube'):
                return self._load_cube_lut(lut_path)
            elif file_ext.endswith('.3dl'):
                return self._load_3dl_lut(lut_path)
            elif file_ext.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.exr')):
                return self._load_image_lut(lut_path)
            else:
                print(f"Unsupported LUT format: {lut_path}")
                print("Supported formats: .cube, .3dl, .png, .jpg, .jpeg, .tiff, .exr")
                return None
        except Exception as e:
            print(f"Error loading LUT file {lut_path}: {e}")
            return None

    def _load_cube_lut(self, cube_path):
        """Load .cube format LUT file with enhanced compatibility"""
        try:
            # Try different encodings for better compatibility
            encodings = ['utf-8', 'latin-1', 'cp1252']
            lines = None
            
            for encoding in encodings:
                try:
                    with open(cube_path, 'r', encoding=encoding) as f:
                        lines = f.readlines()
                    break
                except UnicodeDecodeError:
                    continue
            
            if lines is None:
                print(f"Could not read .cube file with any supported encoding: {cube_path}")
                return None
            
            lut_size = 33  # Default size
            lut_data = []
            domain_min = [0.0, 0.0, 0.0]
            domain_max = [1.0, 1.0, 1.0]
            
            for line in lines:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('#'):
                    continue
                
                # Parse header information
                if line.startswith('LUT_3D_SIZE'):
                    lut_size = int(line.split()[-1])
                    print(f"Loading .cube LUT with size: {lut_size}x{lut_size}x{lut_size}")
                    
                elif line.startswith('DOMAIN_MIN'):
                    domain_min = [float(x) for x in line.split()[1:4]]
                    
                elif line.startswith('DOMAIN_MAX'):
                    domain_max = [float(x) for x in line.split()[1:4]]
                    
                elif line.startswith('TITLE'):
                    title = line.split('"')[1] if '"' in line else line.split()[1]
                    print(f"Loading LUT: {title}")
                    
                # Parse LUT data
                elif not any(line.startswith(keyword) for keyword in ['LUT_1D_SIZE', 'LUT_1D_INPUT_RANGE']):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == 3:
                            # Normalize values if they're outside 0-1 range
                            normalized_values = []
                            for i, val in enumerate(values):
                                if domain_max[i] != domain_min[i]:
                                    normalized = (val - domain_min[i]) / (domain_max[i] - domain_min[i])
                                else:
                                    normalized = val
                                normalized_values.append(max(0.0, min(1.0, normalized)))
                            
                            lut_data.append(normalized_values)
                    except (ValueError, IndexError):
                        continue
            
            # Validate LUT data
            expected_size = lut_size ** 3
            if len(lut_data) == expected_size:
                lut_array = np.array(lut_data, dtype=np.float32).reshape(lut_size, lut_size, lut_size, 3)
                print(f"Successfully loaded .cube LUT: {len(lut_data)} entries")
                return torch.from_numpy(lut_array)
            else:
                print(f"Invalid LUT data size: expected {expected_size}, got {len(lut_data)}")
                
                # Try to handle common size mismatches
                if len(lut_data) > 0:
                    # Find the closest cube root
                    cube_root = round(len(lut_data) ** (1/3))
                    if cube_root ** 3 == len(lut_data):
                        print(f"Adjusting LUT size to {cube_root}x{cube_root}x{cube_root}")
                        lut_array = np.array(lut_data, dtype=np.float32).reshape(cube_root, cube_root, cube_root, 3)
                        return torch.from_numpy(lut_array)
                
                return None
                
        except Exception as e:
            print(f"Error loading .cube file {cube_path}: {e}")
            return None

    def _load_3dl_lut(self, path_3dl):
        """Load .3dl format LUT file"""
        try:
            with open(path_3dl, 'r') as f:
                lines = f.readlines()
            
            lut_data = []
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#'):
                    try:
                        values = [float(x) for x in line.split()]
                        if len(values) == 3:
                            lut_data.append(values)
                    except:
                        continue
            
            # 3dl files are typically 32x32x32
            if len(lut_data) == 32768:  # 32^3
                lut_array = np.array(lut_data, dtype=np.float32).reshape(32, 32, 32, 3)
                print(f"Successfully loaded .3dl LUT: 32x32x32")
                return torch.from_numpy(lut_array)
            else:
                # Try to determine size
                cube_root = round(len(lut_data) ** (1/3))
                if cube_root ** 3 == len(lut_data):
                    lut_array = np.array(lut_data, dtype=np.float32).reshape(cube_root, cube_root, cube_root, 3)
                    print(f"Successfully loaded .3dl LUT: {cube_root}x{cube_root}x{cube_root}")
                    return torch.from_numpy(lut_array)
                else:
                    print(f"Invalid .3dl LUT size: {len(lut_data)} entries")
                    return None
                    
        except Exception as e:
            print(f"Error loading .3dl file {path_3dl}: {e}")
            return None

    def _load_image_lut(self, image_path):
        """Load LUT from image (supports various image formats including Photoshop LUT images)"""
        try:
            lut_image = Image.open(image_path).convert('RGB')
            lut_array = np.array(lut_image, dtype=np.float32) / 255.0
            height, width = lut_array.shape[:2]
            
            print(f"Loading image LUT: {width}x{height}")
            
            # Common LUT image formats:
            # 512x512 for 64^3 LUT (8x8 grid)
            # 1024x32 for 32^3 LUT (32x1 strip)
            # 256x16 for 16^3 LUT (16x1 strip)
            
            # Try different standard formats
            formats = [
                (512, 512, 64),  # 64^3 in 8x8 grid
                (1024, 32, 32),  # 32^3 in horizontal strip
                (256, 16, 16),   # 16^3 in horizontal strip
                (2048, 64, 64),  # 64^3 in horizontal strip
            ]
            
            for expected_w, expected_h, lut_size in formats:
                if width == expected_w and height == expected_h:
                    if expected_w == expected_h:
                        # Square format (grid layout)
                        grid_size = int(lut_size ** (1/3))
                        lut_3d = np.zeros((lut_size, lut_size, lut_size, 3), dtype=np.float32)
                        
                        for r in range(lut_size):
                            for g in range(lut_size):
                                for b in range(lut_size):
                                    # Calculate position in grid
                                    grid_x = (b % grid_size) * lut_size + g
                                    grid_y = (b // grid_size) * lut_size + r
                                    
                                    if grid_x < width and grid_y < height:
                                        lut_3d[r, g, b] = lut_array[grid_y, grid_x]
                    else:
                        # Strip format
                        lut_3d = lut_array.reshape(lut_size, lut_size, lut_size, 3)
                    
                    print(f"Successfully loaded image LUT: {lut_size}x{lut_size}x{lut_size}")
                    return torch.from_numpy(lut_3d)
            
            # If no standard format matches, try to auto-detect
            total_pixels = width * height
            cube_root = round(total_pixels ** (1/3))
            
            if cube_root ** 3 == total_pixels:
                print(f"Auto-detected LUT size: {cube_root}x{cube_root}x{cube_root}")
                lut_3d = lut_array.reshape(cube_root, cube_root, cube_root, 3)
                return torch.from_numpy(lut_3d)
            
            print(f"Could not determine LUT format for {width}x{height} image")
            return None
            
        except Exception as e:
            print(f"Error loading image LUT: {e}")
            return None

    def _apply_lut_tensor(self, images, lut, strength):
        """Apply 3D LUT using tensor operations with trilinear interpolation"""
        batch_size, height, width, channels = images.shape
        lut_size = lut.shape[0]
        
        # Scale input to LUT coordinate space
        coords = images * (lut_size - 1)
        
        # Get integer and fractional parts
        coords_floor = torch.floor(coords).long()
        coords_frac = coords - coords_floor.float()
        
        # Clamp coordinates
        coords_floor = torch.clamp(coords_floor, 0, lut_size - 2)
        coords_ceil = torch.clamp(coords_floor + 1, 0, lut_size - 1)
        
        # Trilinear interpolation
        # Get 8 corner values
        c000 = lut[coords_floor[:, :, :, 0], coords_floor[:, :, :, 1], coords_floor[:, :, :, 2]]
        c001 = lut[coords_floor[:, :, :, 0], coords_floor[:, :, :, 1], coords_ceil[:, :, :, 2]]
        c010 = lut[coords_floor[:, :, :, 0], coords_ceil[:, :, :, 1], coords_floor[:, :, :, 2]]
        c011 = lut[coords_floor[:, :, :, 0], coords_ceil[:, :, :, 1], coords_ceil[:, :, :, 2]]
        c100 = lut[coords_ceil[:, :, :, 0], coords_floor[:, :, :, 1], coords_floor[:, :, :, 2]]
        c101 = lut[coords_ceil[:, :, :, 0], coords_floor[:, :, :, 1], coords_ceil[:, :, :, 2]]
        c110 = lut[coords_ceil[:, :, :, 0], coords_ceil[:, :, :, 1], coords_floor[:, :, :, 2]]
        c111 = lut[coords_ceil[:, :, :, 0], coords_ceil[:, :, :, 1], coords_ceil[:, :, :, 2]]
        
        # Interpolate
        xf, yf, zf = coords_frac[:, :, :, 0:1], coords_frac[:, :, :, 1:2], coords_frac[:, :, :, 2:3]
        
        c00 = c000 * (1 - xf) + c100 * xf
        c01 = c001 * (1 - xf) + c101 * xf
        c10 = c010 * (1 - xf) + c110 * xf
        c11 = c011 * (1 - xf) + c111 * xf
        
        c0 = c00 * (1 - yf) + c10 * yf
        c1 = c01 * (1 - yf) + c11 * yf
        
        result_lut = c0 * (1 - zf) + c1 * zf
        
        # Blend with original based on strength
        result = images * (1 - strength) + result_lut * strength
        
        return torch.clamp(result, 0, 1)
