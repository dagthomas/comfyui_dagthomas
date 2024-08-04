# Converting the prompt generation script into a ComfyUI plugin structure
import random
import json
import requests
import comfy.sd
import comfy.model_management
import nodes
import torch
import torchvision.transforms as transforms
import os
import base64
from io import BytesIO
from openai import OpenAI
import torch
import numpy as np
from PIL import Image
from datetime import datetime

# Function to load data from a JSON file
def load_json_file(file_name):
    # Construct the absolute path to the data file
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)


# import nodes
import re

ARTFORM = load_json_file("artform.json")
PHOTO_FRAMING = load_json_file("photo_framing.json")
PHOTO_TYPE = load_json_file("photo_type.json")
DEFAULT_TAGS = load_json_file("default_tags.json")
ROLES = load_json_file("roles.json")
HAIRSTYLES = load_json_file("hairstyles.json")
ADDITIONAL_DETAILS = load_json_file("additional_details.json")
PHOTOGRAPHY_STYLES = load_json_file("photography_styles.json")
DEVICE = load_json_file("device.json")
PHOTOGRAPHER = load_json_file("photographer.json")
ARTIST = load_json_file("artist.json")
DIGITAL_ARTFORM = load_json_file("digital_artform.json")
PLACE = load_json_file("place.json")
LIGHTING = load_json_file("lighting.json")
CLOTHING = load_json_file("clothing.json")
COMPOSITION = load_json_file("composition.json")
POSE = load_json_file("pose.json")
BACKGROUND = load_json_file("background.json")
BODY_TYPES = load_json_file("body_types.json")
CUSTOM_CATEGORY = "comfyui_dagthomas"
@torch.no_grad()
def skimmed_CFG(x_orig, cond, uncond, cond_scale, skimming_scale):
    denoised = x_orig - ((x_orig - uncond) + cond_scale * (cond - uncond))
    matching_pred_signs = torch.sign(cond - uncond) == torch.sign(cond)
    matching_diff_after = torch.sign(cond) == torch.sign(cond * cond_scale - uncond * (cond_scale - 1))
    deviation_influence = torch.sign(denoised) == torch.sign(denoised - x_orig)
    outer_influence = matching_pred_signs & matching_diff_after & deviation_influence
    low_cfg_denoised_outer = x_orig - ((x_orig - uncond) + skimming_scale * (cond - uncond))
    low_cfg_denoised_outer_difference = denoised - low_cfg_denoised_outer
    cond[outer_influence] -= low_cfg_denoised_outer_difference[outer_influence] / cond_scale
    
    return cond
class RandomIntegerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("INT", {
                    "default": 0, 
                    "min": -1000000000, 
                    "max": 1000000000,
                    "step": 1
                }),
                "max_value": ("INT", {
                    "default": 10, 
                    "min": -1000000000, 
                    "max": 1000000000,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_random_int"
    CATEGORY = CUSTOM_CATEGORY

    def generate_random_int(self, min_value, max_value):
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return (random.randint(min_value, max_value),)
    
class FlexibleStringMergerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
            },
            "optional": {
                "string2": ("STRING", {"default": ""}),
                "string3": ("STRING", {"default": ""}),
                "string4": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = CUSTOM_CATEGORY

    def merge_strings(self, string1, string2="", string3="", string4=""):
        def process_input(s):
            if isinstance(s, list):
                return ", ".join(str(item) for item in s)
            return str(s).strip()

        strings = [process_input(s) for s in [string1, string2, string3, string4] if process_input(s)]
        if not strings:
            return ("")  # Return an empty string if no non-empty inputs
        return (" AND ".join(strings),)

class StringMergerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": ""}),
                "string2": ("STRING", {"default": ""})
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = "string_operations"

    def merge_strings(self, string1, string2):
        def process_input(s):
            if isinstance(s, list):
                return ", ".join(str(item) for item in s)
            return str(s)

        processed_string1 = process_input(string1)
        processed_string2 = process_input(string2)
        merged = f"{processed_string1} AND {processed_string2}"
        return (merged,)
    
class CFGSkimmingSingleScalePreCFGNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "skimming_cfg": ("FLOAT", {
                    "default": 7.0, 
                    "min": 0.0, 
                    "max": 7.0, 
                    "step": 0.1, 
                    "round": 0.01
                }),
                "razor_skim": ("BOOLEAN", {"default": False})
            }
        }
    
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = CUSTOM_CATEGORY

    def patch(self, model, skimming_cfg, razor_skim):
        @torch.no_grad()
        def pre_cfg_patch(args):
            conds_out, cond_scale, x_orig = args["conds_out"], args["cond_scale"], args['input']
            
            if not torch.any(conds_out[1]):
                return conds_out
            
            uncond_skimming_scale = 0 if razor_skim else skimming_cfg
            conds_out[1] = skimmed_CFG(x_orig, conds_out[1], conds_out[0], cond_scale, uncond_skimming_scale)
            conds_out[0] = skimmed_CFG(x_orig, conds_out[0], conds_out[1], cond_scale, skimming_cfg)
            
            return conds_out

        new_model = model.clone()
        new_model.set_model_sampler_pre_cfg_function(pre_cfg_patch)
        return (new_model,)
    
class PGSD3LatentGenerator:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "width": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "height": ("INT", {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8}),
                              "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})}}
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def generate(self, width=1024, height=1024, batch_size=1):
        def adjust_dimensions(width, height):
            megapixel = 1_000_000
            multiples = 64

            if width == 0 and height != 0:
                height = int(height)
                width = megapixel // height
                
                if width % multiples != 0:
                    width += multiples - (width % multiples)
                
                if width * height > megapixel:
                    width -= multiples

            elif height == 0 and width != 0:
                width = int(width)
                height = megapixel // width
                
                if height % multiples != 0:
                    height += multiples - (height % multiples)
                
                if width * height > megapixel:
                    height -= multiples

            elif width == 0 and height == 0:
                width = 1024
                height = 1024

            return width, height

        width, height = adjust_dimensions(width, height)

        latent = torch.ones([batch_size, 16, height // 8, width // 8], device=self.device) * 0.0609
        return ({"samples": latent}, )
    
class GPT4VisionNode:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def tensor_to_pil(self, img_tensor):
        i = 255. * img_tensor.cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)
    
    def save_prompt(self, prompt):
        filename_text = "vision_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        # Save the prompt to the file
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def analyze_images(self, images, happy_talk, compress, compression_level, custom_base_prompt=""):
        try:
            default_happy_prompt = """Analyze the provided images and create a detailed visually descriptive caption that combines elements from all images into a single cohesive composition. This caption will be used as a prompt for a text-to-image AI system. Focus on:
1. Detailed visual descriptions of characters, including ethnicity, skin tone, expressions, etc.
2. Overall scene and background details.
3. Image style, photographic techniques, direction of photo taken.
4. Cinematography aspects with technical details.
5. If multiple characters are present, describe an interesting interaction between two primary characters.
7. Describe the lighting setup in detail, including type, color, and placement of light sources.

Examples of prompts to generate: 

1. Ethereal cyborg woman, bioluminescent jellyfish headdress. Steampunk goggles blend with translucent tentacles. Cracked porcelain skin meets iridescent scales. Mechanical implants and delicate tendrils intertwine. Human features with otherworldly glow. Dreamy aquatic hues contrast weathered metal. Reflective eyes capture unseen worlds. Soft bioluminescence meets harsh desert backdrop. Fusion of organic and synthetic, ancient and futuristic. Hyper-detailed textures, surreal atmosphere.

2. Photo of a broken ruined cyborg girl in a landfill, robot, body is broken with scares and holes,half the face is android,laying on the ground, creating a hyperpunk scene with desaturated dark red and blue details, colorful polaroid with vibrant colors, (vacations, high resolution:1.3), (small, selective focus, european film:1.2)

3. Horror-themed (extreme close shot of eyes :1.3) of nordic woman, (war face paint:1.2), mohawk blonde haircut wit thin braids, runes tattoos, sweat, (detailed dirty skin:1.3) shiny, (epic battleground backgroun :1.2), . analog, haze, ( lens blur :1.3) , hard light, sharp focus on eyes, low saturation

ALWAYS remember to out that it is a movie still and describe the film grain, color grading, and any artifacts or characteristics specific to film photography.
ALWAYS create the output as one scene, never transition between scenes.
"""

            default_simple_prompt = """Analyze the provided images and create a brief, straightforward caption that combines key elements from all images. Focus on the main subjects, overall scene, and atmosphere. Provide a clear and concise description in one or two sentences, suitable for a text-to-image AI system."""

            if custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

            if compress:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            messages = [{"role": "user", "content": [{"type": "text", "text": base_prompt}]}]

            # Process each image in the batch
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                base64_image = self.encode_image(pil_image)
                messages[0]["content"].append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                })

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages
            )
            self.save_prompt(response.choices[0].message.content)
            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Images tensor shape: {images.shape}")
            print(f"Images tensor type: {images.dtype}")
            return (f"Error occurred while processing the request: {str(e)}",)
        
class RandomIntegerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": ("INT", {
                    "default": 0, 
                    "min": -1000000000, 
                    "max": 1000000000,
                    "step": 1
                }),
                "max_value": ("INT", {
                    "default": 10, 
                    "min": -1000000000, 
                    "max": 1000000000,
                    "step": 1
                })
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_random_int"
    CATEGORY = CUSTOM_CATEGORY

    def generate_random_int(self, min_value, max_value):
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        return (random.randint(min_value, max_value),)
            
class OllamaNode:
    def __init__(self):
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_model": ("STRING", {"default": "llama3.1:8b"}),
                "ollama_url": ("STRING", {"default": "http://localhost:11434/api/generate"})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def save_prompt(self, prompt):
        filename_text = "mini_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def generate(self, input_text, happy_talk, compress, compression_level, custom_base_prompt="", custom_model="llama3.1:8b", ollama_url="http://localhost:11434/api/generate"):
        try:
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene as a screen grab from a 35mm film still, including details about the film grain, color grading, and any artifacts or characteristics specific to 35mm film photography."""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            base_prompt = custom_base_prompt.strip() if custom_base_prompt.strip() else (default_happy_prompt if happy_talk else default_simple_prompt)

            if compress:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be more concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            prompt = f"{base_prompt}\nDescription: {input_text}"

            payload = {
                "model": custom_model,
                "prompt": prompt,
                "stream": False
            }

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()['response']

            self.save_prompt(result)
            return (result,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)
                
class GPT4MiniNode:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_text": ("STRING", {"multiline": True}),
                "happy_talk": ("BOOLEAN", {"default": True}),
                "compress": ("BOOLEAN", {"default": False}),
                "compression_level": (["soft", "medium", "hard"],),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""})
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def save_prompt(self, prompt):
        filename_text = "mini_" + prompt.split(',')[0].strip()
        filename_text = re.sub(r'[^\w\-_\. ]', '_', filename_text)
        filename_text = filename_text[:30]  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)
        
        # Save the prompt to the file
        with open(filename, "w") as file:
            file.write(prompt)
        
        print(f"Prompt saved to {filename}")

    def generate(self, input_text, happy_talk, compress, compression_level, custom_base_prompt=""):
        try:
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene as a screen grab from a 35mm film still, including details about the film grain, color grading, and any artifacts or characteristics specific to 35mm film photography."""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            if custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

            if compress and happy_talk:
                compression_chars = {
                    "soft": 600,
                    "medium": 400,
                    "hard": 200
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be more concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."
            elif compress and not happy_talk:
                compression_chars = {
                    "soft": 300,
                    "medium": 200,
                    "hard": 100
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Limit the response to no more than {char_limit} characters."

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{base_prompt}\nDescription: {input_text}"}],
            )
            
            self.save_prompt(response.choices[0].message.content)
            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)
         
class PromptGenerator:
    RETURN_TYPES = (
        "STRING",
        "INT",
        "STRING",
        "STRING",
        "STRING"
    )
    RETURN_NAMES = (
        "prompt",
        "seed",
        "t5xxl",
        "clip_l",
        "clip_g",
    )
    FUNCTION = "generate_prompt"
    CATEGORY = CUSTOM_CATEGORY

    def __init__(self, seed=None):
        self.rng = random.Random(seed)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": (
                    "INT",
                    {"default": 0, "min": 0, "max": 1125899906842624},
                ),
                "custom": ("STRING", {}),
                "subject": ("STRING", {}),
                "artform": (
                    ["disabled"] + ["random"] + ARTFORM,
                    {"default": "photography"},
                ),
                "photo_type": (
                    ["disabled"] + ["random"] + PHOTO_TYPE,
                    {"default": "random"},
                ),
                "body_types": (
                    ["disabled"] + ["random"] + BODY_TYPES,
                    {"default": "random"},
                ),
                "default_tags": (
                    ["disabled"] + ["random"] + DEFAULT_TAGS,
                    {"default": "random"},
                ),
                "roles": (["disabled"] + ["random"] + ROLES, {"default": "random"}),
                "hairstyles": (
                    ["disabled"] + ["random"] + HAIRSTYLES,
                    {"default": "random"},
                ),
                "additional_details": (
                    ["disabled"] + ["random"] + ADDITIONAL_DETAILS,
                    {"default": "random"},
                ),
                "photography_styles": (
                    ["disabled"] + ["random"] + PHOTOGRAPHY_STYLES,
                    {"default": "random"},
                ),
                "device": (["disabled"] + ["random"] + DEVICE, {"default": "random"}),
                "photographer": (
                    ["disabled"] + ["random"] + PHOTOGRAPHER,
                    {"default": "random"},
                ),
                "artist": (["disabled"] + ["random"] + ARTIST, {"default": "random"}),
                "digital_artform": (
                    ["disabled"] + ["random"] + DIGITAL_ARTFORM,
                    {"default": "random"},
                ),
                "place": (["disabled"] + ["random"] + PLACE, {"default": "random"}),
                "lighting": (
                    ["disabled"] + ["random"] + LIGHTING,
                    {"default": "random"},
                ),
                "clothing": (
                    ["disabled"] + ["random"] + CLOTHING,
                    {"default": "random"},
                ),
                "composition": (
                    ["disabled"] + ["random"] + COMPOSITION,
                    {"default": "random"},
                ),
                "pose": (
                    ["disabled"] + ["random"] + POSE,
                    {"default": "random"},
                ),
                "background": (
                    ["disabled"] + ["random"] + BACKGROUND,
                    {"default": "random"},
                ),
            },
        }

    def split_and_choose(self, input_str):
        choices = [choice.strip() for choice in input_str.split(",")]
        return self.rng.choices(choices, k=1)[0]

    def get_choice(self, input_str, default_choices):
        if input_str.lower() == "disabled":
            return ""
        elif "," in input_str:
            return self.split_and_choose(input_str)
        elif input_str.lower() == "random":
            return self.rng.choices(default_choices, k=1)[0]
        else:
            return input_str
    def clean_consecutive_commas(self, input_string):
        cleaned_string = re.sub(r',\s*,', ',', input_string)
        return cleaned_string
    
    def process_string(self, replaced, seed):
        replaced = re.sub(r'\s*,\s*', ',', replaced)
        replaced = re.sub(r',+', ',', replaced)
        original = replaced
        
        # Find the indices for "BREAK_CLIPL"
        first_break_clipl_index = replaced.find("BREAK_CLIPL")
        second_break_clipl_index = replaced.find("BREAK_CLIPL", first_break_clipl_index + len("BREAK_CLIPL"))
        
        # Extract the content between "BREAK_CLIPL" markers
        if first_break_clipl_index != -1 and second_break_clipl_index != -1:
            clip_content_l = replaced[first_break_clipl_index + len("BREAK_CLIPL"):second_break_clipl_index]
            
            # Update the replaced string by removing the "BREAK_CLIPL" content
            replaced = replaced[:first_break_clipl_index].strip(", ") + replaced[second_break_clipl_index + len("BREAK_CLIPL"):].strip(", ")
            
            clip_l = clip_content_l
        else:
            clip_l = ""
        
        # Find the indices for "BREAK_CLIPG"
        first_break_clipg_index = replaced.find("BREAK_CLIPG")
        second_break_clipg_index = replaced.find("BREAK_CLIPG", first_break_clipg_index + len("BREAK_CLIPG"))
        
        # Extract the content between "BREAK_CLIPG" markers
        if first_break_clipg_index != -1 and second_break_clipg_index != -1:
            clip_content_g = replaced[first_break_clipg_index + len("BREAK_CLIPG"):second_break_clipg_index]
            
            # Update the replaced string by removing the "BREAK_CLIPG" content
            replaced = replaced[:first_break_clipg_index].strip(", ") + replaced[second_break_clipg_index + len("BREAK_CLIPG"):].strip(", ")
            
            clip_g = clip_content_g
        else:
            clip_g = ""
        
        t5xxl = replaced
        
        original = original.replace("BREAK_CLIPL", "").replace("BREAK_CLIPG", "")
        original = re.sub(r'\s*,\s*', ',', original)
        original = re.sub(r',+', ',', original)
        clip_l = re.sub(r'\s*,\s*', ',', clip_l)
        clip_l = re.sub(r',+', ',', clip_l)
        clip_g = re.sub(r'\s*,\s*', ',', clip_g)
        clip_g = re.sub(r',+', ',', clip_g)
        if clip_l.startswith(","):
            clip_l = clip_l[1:]
        if clip_g.startswith(","):
            clip_g = clip_g[1:]
        if original.startswith(","):
            original = original[1:]
        if t5xxl.startswith(","):
            t5xxl = t5xxl[1:]

        # print(f"PromptGenerator String: {replaced}")
        print(f"prompt: {original}")
        print("")
        print(f"clip_l: {clip_l}")
        print("")
        print(f"clip_g: {clip_g}")
        print("")
        print(f"t5xxl: {t5xxl}")
        print("")

        return original, seed, t5xxl, clip_l, clip_g
    
    def generate_prompt(self, **kwargs):
        seed = kwargs.get("seed", 0)
        
        if seed is not None:
            self.rng = random.Random(seed)
        components = []
        custom = kwargs.get("custom", "")
        if custom:
            components.append(custom)
        is_photographer = kwargs.get("artform", "").lower() == "photography" or (
            kwargs.get("artform", "").lower() == "random"
            and self.rng.choice([True, False])
        )




        subject = kwargs.get("subject", "")

        if is_photographer:
            selected_photo_style = self.get_choice(
                kwargs.get("photography_styles", ""), PHOTOGRAPHY_STYLES
            )
            if not selected_photo_style:
                selected_photo_style = "photography"
            components.append(selected_photo_style)
            if kwargs.get("photography_style", "") != "disabled" and kwargs.get("default_tags", "") != "disabled" or subject != "":
                components.append(" of")
        
        
        default_tags = kwargs.get(
            "default_tags", "random"
        )  # default to "random" if not specified
        body_type = kwargs.get("body_types", "")
        if not subject:
            if default_tags == "random":
                # Case where default_tags is "random"
                

                # Check if body_types is neither "disabled" nor "random"
                if body_type != "disabled" and body_type != "random":
                    selected_subject = self.get_choice(
                        kwargs.get("default_tags", ""), DEFAULT_TAGS
                    ).replace("a ", "").replace("an ", "")
                    components.append("a ")
                    components.append(body_type)
                    components.append(selected_subject)
                elif body_type == "disabled":
                    selected_subject = self.get_choice(
                        kwargs.get("default_tags", ""), DEFAULT_TAGS
                    )
                    components.append(selected_subject)
                else:
                    # When body_types is "disabled" or "random"
                    body_type = self.get_choice(body_type, BODY_TYPES)
                    components.append("a ")
                    components.append(body_type)
                    selected_subject = self.get_choice(
                        kwargs.get("default_tags", ""), DEFAULT_TAGS
                    ).replace("a ", "").replace("an ", "")
                    components.append(selected_subject)
            elif default_tags == "disabled":
                # Do nothing if default_tags is "disabled"
                pass
            else:
                # Add default_tags if it's not "random" or "disabled"
                components.append(default_tags)
        else:
            if body_type != "disabled" and body_type != "random":
                components.append("a ")
                components.append(body_type)
            elif body_type == "disabled":
                pass
            else:
                body_type = self.get_choice(body_type, BODY_TYPES)
                components.append("a ")
                components.append(body_type)

            components.append(subject)


        params = [
            ("roles", ROLES),
            ("hairstyles", HAIRSTYLES),
            ("additional_details", ADDITIONAL_DETAILS),
            
        ]
        for param in params:
            components.append(self.get_choice(kwargs.get(param[0], ""), param[1]))
        for i in reversed(range(len(components))):
            if components[i] in PLACE:
                components[i] += ","
                break
        if kwargs.get("clothing", "") != "disabled" and kwargs.get("clothing", "") != "random":
            components.append(", dressed in ")
            clothing = kwargs.get("clothing", "")
            components.append(clothing)
        elif kwargs.get("clothing", "") == "random":
            components.append(", dressed in ")
            clothing = self.get_choice(
                    kwargs.get("clothing", ""), CLOTHING
                )
            components.append(clothing)

        if kwargs.get("composition", "") != "disabled" and kwargs.get("composition", "") != "random":
            components.append(",")
            composition = kwargs.get("composition", "")
            components.append(composition)
        elif kwargs.get("composition", "") == "random": 
            components.append(",")
            composition = self.get_choice(
                    kwargs.get("composition", ""), COMPOSITION
                )
            components.append(composition)
        
        if kwargs.get("pose", "") != "disabled" and kwargs.get("pose", "") != "random":
            components.append(",")
            pose = kwargs.get("pose", "")
            components.append(pose)
        elif kwargs.get("pose", "") == "random":
            components.append(",")
            pose = self.get_choice(
                    kwargs.get("pose", ""), POSE
                )
            components.append(pose)
        components.append("BREAK_CLIPG")
        if kwargs.get("background", "") != "disabled" and kwargs.get("background", "") != "random":
            components.append(",")
            background = kwargs.get("background", "")
            components.append(background)
        elif kwargs.get("background", "") == "random": 
            components.append(",")
            background = self.get_choice(
                    kwargs.get("background", ""), BACKGROUND
                )
            components.append(background)

        if kwargs.get("place", "") != "disabled" and kwargs.get("place", "") != "random":
            components.append(",")
            place = kwargs.get("place", "")
            components.append(place)
        elif kwargs.get("place", "") == "random": 
            components.append(",")
            place = self.get_choice(
                    kwargs.get("place", ""), PLACE
                )
            components.append(place + ",")


        
        lighting = kwargs.get("lighting", "").lower()
        if lighting == "random":
            
            selected_lighting = ", ".join(
                self.rng.sample(LIGHTING, self.rng.randint(2, 5))
            )
            components.append(",")
            components.append(selected_lighting)
        elif lighting == "disabled":
            pass
        else:
            components.append(", ")
            components.append(lighting)
        components.append("BREAK_CLIPG")
        components.append("BREAK_CLIPL")
        if is_photographer:
            if kwargs.get("photo_type", "") != "disabled":
                photo_type_choice = self.get_choice(
                    kwargs.get("photo_type", ""), PHOTO_TYPE
                )
                if photo_type_choice and photo_type_choice != "random" and photo_type_choice != "disabled":
                    random_value = round(self.rng.uniform(1.1, 1.5), 1)
                    components.append(f", ({photo_type_choice}:{random_value}), ")
            

            params = [
                # ("photography_styles", PHOTOGRAPHY_STYLES),
                ("device", DEVICE),
                ("photographer", PHOTOGRAPHER),
            ]
            components.extend(
                [
                    self.get_choice(kwargs.get(param[0], ""), param[1])
                    for param in params
                ]
            )
            if kwargs.get("device", "") != "disabled":
                components[-2] = f", shot on {components[-2]}"
            if kwargs.get("photographer", "") != "disabled":
                components[-1] = f", photo by {components[-1]}"
        else:
            digital_artform_choice = self.get_choice(
                kwargs.get("digital_artform", ""), DIGITAL_ARTFORM
            )
            if digital_artform_choice:
                components.append(f"{digital_artform_choice}")
            if kwargs.get("artist", "") != "disabled":
                components.append(f"by {self.get_choice(kwargs.get('artist', ''), ARTIST)}")
        components.append("BREAK_CLIPL")

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        print(f"PromptGenerator Seed  : {seed}")
        replaced = prompt.replace("of as", "of")
        replaced = self.clean_consecutive_commas(replaced)
        


        return self.process_string(replaced, seed)


NODE_CLASS_MAPPINGS = {
    "RandomIntegerNode": RandomIntegerNode,
    "OllamaNode": OllamaNode,
    "FlexibleStringMergerNode": FlexibleStringMergerNode,
    "StringMergerNode": StringMergerNode,
    "CFGSkimming": CFGSkimmingSingleScalePreCFGNode,
    "GPT4VisionNode": GPT4VisionNode,
    "GPT4MiniNode": GPT4MiniNode,
    "PromptGenerator": PromptGenerator,
    "PGSD3LatentGenerator": PGSD3LatentGenerator,
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "RandomIntegerNode": "Random Integer Generator",
    "GPT4MiniNode": "LLM morbuto generator",
    "PromptGenerator": "Auto Prompter",
    "PGSD3LatentGenerator": "PGSD3LatentGenerator", 
    "GPT4VisionNode": "GPT4VisionNode",
    "CFGSkimming": "CFG Skimming",
    "StringMergerNode": "String Merger", 
    "FlexibleStringMergerNode": "Flexible String Merger",
    "OllamaNode": "OllamaNode",
}


