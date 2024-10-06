# Converting the prompt generation script into a ComfyUI plugin structure
import random
import json
import requests
import comfy.sd
import comfy.model_management
import nodes
import torch
import os
import google.generativeai as genai
import base64
from io import BytesIO
from openai import OpenAI
import torch
import numpy as np
from datetime import datetime
import codecs
from PIL import Image
import chardet
import io
import folder_paths      
from transformers import AutoModelForCausalLM
from transformers import AutoProcessor 

def tensor2pil(t_image: torch.Tensor)  -> Image:
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))


# Function to load data from a JSON file
def load_json_file(file_name):
    # Construct the absolute path to the data file
    file_path = os.path.join(os.path.dirname(__file__), "data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)


def load_all_json_files(base_path):
    data = {}
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, base_path)
                key = os.path.splitext(relative_path)[0].replace(os.path.sep, "_")
                try:
                    with codecs.open(file_path, "r", "utf-8") as f:
                        data[key] = json.load(f)
                except UnicodeDecodeError:
                    print(
                        f"Warning: Unable to decode file {file_path} with UTF-8 encoding. Skipping this file."
                    )
                except json.JSONDecodeError:
                    print(
                        f"Warning: Invalid JSON in file {file_path}. Skipping this file."
                    )
    return data


# Assuming your script is in the same directory as the 'data' folder
base_dir = os.path.dirname(__file__)
next_dir = os.path.join(base_dir, "data", "next")
prompt_dir = os.path.join(base_dir, "data", "custom_prompts")
# custom_dir = os.path.join(base_dir, "data", "custom")
# Load all JSON files
all_data = load_all_json_files(next_dir)

# Now you can access the data using keys like:
# all_data['brands']
# all_data['architecture_architect']
# all_data['art_painting']
# etc.


# print(all_data.keys())

# To access a specific file's data:
# if 'brands' in all_data:
#     print(all_data['brands'])

# if 'architecture_architect' in all_data:
#     print(all_data['architecture_architect'])

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


class DynamicStringCombinerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_inputs": (["1", "2", "3", "4", "5"],),
                "user_text": ("STRING", {"multiline": True}),
            },
            "optional": {
                "string1": ("STRING", {"multiline": False}),
                "string2": ("STRING", {"multiline": False}),
                "string3": ("STRING", {"multiline": False}),
                "string4": ("STRING", {"multiline": False}),
                "string5": ("STRING", {"multiline": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "combine_strings"
    CATEGORY = CUSTOM_CATEGORY

    def combine_strings(
        self,
        num_inputs,
        user_text,
        string1="",
        string2="",
        string3="",
        string4="",
        string5="",
    ):
        # Convert num_inputs to integer
        n = int(num_inputs)

        # Get the specified number of input strings
        input_strings = [string1, string2, string3, string4, string5][:n]

        # Combine the input strings
        combined = ", ".join(s for s in input_strings if s.strip())

        # Append the user_text to the result
        result = f"{combined}\nUser Input: {user_text}"

        return (result,)

    @classmethod
    def IS_CHANGED(
        s, num_inputs, user_text, string1, string2, string3, string4, string5
    ):
        return float(num_inputs)


class SentenceMixerNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input1": ("STRING", {"multiline": True}),
            },
            "optional": {
                "input2": ("STRING", {"multiline": True}),
                "input3": ("STRING", {"multiline": True}),
                "input4": ("STRING", {"multiline": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "mix_sentences"
    CATEGORY = CUSTOM_CATEGORY

    def mix_sentences(self, input1, input2="", input3="", input4=""):
        def process_input(input_data):
            if isinstance(input_data, list):
                return " ".join(input_data)
            return input_data

        all_text = " ".join(
            filter(
                bool,
                [process_input(input) for input in [input1, input2, input3, input4]],
            )
        )

        sentences = []
        current_sentence = ""
        for char in all_text:
            current_sentence += char
            if char in [".", ","]:
                sentences.append(current_sentence.strip())
                current_sentence = ""
        if current_sentence:
            sentences.append(current_sentence.strip())

        random.shuffle(sentences)

        result = " ".join(sentences)

        return (result,)


# This line is needed to register the node in ComfyUI
NODE_CLASS_MAPPINGS = {"SentenceMixerNode": SentenceMixerNode}


class RandomIntegerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": (
                    "INT",
                    {"default": 0, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
                "max_value": (
                    "INT",
                    {"default": 10, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_random_int"
    CATEGORY = CUSTOM_CATEGORY  # Replace with your actual category

    def generate_random_int(self, min_value, max_value):
        if min_value > max_value:
            min_value, max_value = max_value, min_value
        result = random.randint(min_value, max_value)
        # print(f"Generating random int between {min_value} and {max_value}: {result}")
        return (result,)


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
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = CUSTOM_CATEGORY

    def merge_strings(self, string1, string2="", string3="", string4=""):
        def process_input(s):
            if isinstance(s, list):
                return ", ".join(str(item) for item in s)
            return str(s).strip()

        strings = [
            process_input(s)
            for s in [string1, string2, string3, string4]
            if process_input(s)
        ]
        if not strings:
            return ""  # Return an empty string if no non-empty inputs
        return (" AND ".join(strings),)


class StringMergerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string1": ("STRING", {"default": "", "forceInput": True}),
                "string2": ("STRING", {"default": "", "forceInput": True}),
                "use_and": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "merge_strings"
    CATEGORY = CUSTOM_CATEGORY

    def merge_strings(self, string1, string2, use_and):
        def process_input(s):
            if isinstance(s, list):
                return ",".join(str(item).strip() for item in s)
            return str(s).strip()

        processed_string1 = process_input(string1)
        processed_string2 = process_input(string2)
        separator = " AND " if use_and else ","
        merged = f"{processed_string1}{separator}{processed_string2}"

        # Remove double commas and clean spaces around commas
        merged = merged.replace(",,", ",").replace(" ,", ",").replace(", ", ",")

        # Clean leading and trailing spaces
        merged = merged.strip()

        return (merged,)

class PGSD3LatentGenerator:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

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

        latent = (
            torch.ones([batch_size, 16, height // 8, width // 8], device=self.device)
            * 0.0609
        )
        return ({"samples": latent},)


class APNLatent:
    def __init__(self):
        self.device = comfy.model_management.intermediate_device()

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "width": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "height": (
                    "INT",
                    {"default": 1024, "min": 0, "max": nodes.MAX_RESOLUTION, "step": 8},
                ),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096}),
                "megapixel_scale": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1},
                ),
                "aspect_ratio": (
                    ["1:1", "3:2", "4:3", "16:9", "21:9"],
                    {"default": "1:1"},
                ),
                "is_portrait": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT", "INT", "INT")
    RETURN_NAMES = ("LATENT", "width", "height")
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def generate(
        self,
        width=1024,
        height=1024,
        batch_size=1,
        megapixel_scale=1.0,
        aspect_ratio="1:1",
        is_portrait=False,
    ):
        def adjust_dimensions(megapixels, aspect_ratio, is_portrait):
            aspect_ratios = {
                "1:1": (1, 1),
                "3:2": (3, 2),
                "4:3": (4, 3),
                "16:9": (16, 9),
                "21:9": (21, 9),
            }

            ar_width, ar_height = aspect_ratios[aspect_ratio]
            if is_portrait:
                ar_width, ar_height = ar_height, ar_width

            total_pixels = int(megapixels * 1_000_000)

            width = int((total_pixels * ar_width / ar_height) ** 0.5)
            height = int(total_pixels / width)

            # Round to nearest multiple of 64
            width = (width + 32) // 64 * 64
            height = (height + 32) // 64 * 64

            return width, height

        if width == 0 or height == 0:
            width, height = adjust_dimensions(
                megapixel_scale, aspect_ratio, is_portrait
            )
        else:
            # If width and height are provided, adjust them to fit within the megapixel scale
            current_mp = (width * height) / 1_000_000
            if current_mp > megapixel_scale:
                scale_factor = (megapixel_scale / current_mp) ** 0.5
                width = int((width * scale_factor + 32) // 64 * 64)
                height = int((height * scale_factor + 32) // 64 * 64)

            # Swap width and height if portrait is selected and current orientation doesn't match
            if is_portrait and width > height:
                width, height = height, width
            elif not is_portrait and height > width:
                width, height = height, width

        latent = (
            torch.ones([batch_size, 16, height // 8, width // 8], device=self.device)
            * 0.0609
        )
        return ({"samples": latent}, width, height)


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
                "poster": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),  # New override field
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def encode_image(self, image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def tensor_to_pil(self, img_tensor):
        i = 255.0 * img_tensor.cpu().numpy()
        img_array = np.clip(i, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def save_prompt(self, prompt):
        filename_text = "vision_" + prompt.split(",")[0].strip()
        filename_text = re.sub(r"[^\w\-_\. ]", "_", filename_text)
        filename_text = filename_text[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        with open(filename, "w") as file:
            file.write(prompt)

        print(f"Prompt saved to {filename}")

    def analyze_images(
        self,
        images,
        happy_talk,
        compress,
        compression_level,
        poster,
        custom_base_prompt="",
        custom_title="",
        override="",
    ):
        try:
            default_happy_prompt = """Analyze the provided images and create a detailed visually descriptive caption that combines elements from all images into a single cohesive composition.Imagine all images being movie stills from real movies. This caption will be used as a prompt for a text-to-image AI system. Focus on:
1. Detailed visual descriptions of characters, including ethnicity, skin tone, expressions, etc.
2. Overall scene and background details.
3. Image style, photographic techniques, direction of photo taken.
4. Cinematography aspects with technical details.
5. If multiple characters are present, describe an interesting interaction between two primary characters.
6. Incorporate a specific movie director's visual style (e.g., Wes Anderson, Christopher Nolan, Quentin Tarantino).
7. Describe the lighting setup in detail, including type, color, and placement of light sources.

If you want to add text, state the font, style and where it should be placed, a sign, a poster, etc. The text should be captioned like this " ".
Always describn how characters look at each other, their expressions, and the overall mood of the scene.
Examples of prompts to generate: 

1. Ethereal cyborg woman, bioluminescent jellyfish headdress. Steampunk goggles blend with translucent tentacles. Cracked porcelain skin meets iridescent scales. Mechanical implants and delicate tendrils intertwine. Human features with otherworldly glow. Dreamy aquatic hues contrast weathered metal. Reflective eyes capture unseen worlds. Soft bioluminescence meets harsh desert backdrop. Fusion of organic and synthetic, ancient and futuristic. Hyper-detailed textures, surreal atmosphere.

2. Photo of a broken ruined cyborg girl in a landfill, robot, body is broken with scares and holes,half the face is android,laying on the ground, creating a hyperpunk scene with desaturated dark red and blue details, colorful polaroid with vibrant colors, (vacations, high resolution:1.3), (small, selective focus, european film:1.2)

3. Horror-themed (extreme close shot of eyes :1.3) of nordic woman, (war face paint:1.2), mohawk blonde haircut wit thin braids, runes tattoos, sweat, (detailed dirty skin:1.3) shiny, (epic battleground backgroun :1.2), . analog, haze, ( lens blur :1.3) , hard light, sharp focus on eyes, low saturation

ALWAYS remember to out that it is a cinematic movie still and describe the film grain, color grading, and any artifacts or characteristics specific to film photography.
ALWAYS create the output as one scene, never transition between scenes.
"""

            default_simple_prompt = """Analyze the provided images and create a brief, straightforward caption that combines key elements from all images. Focus on the main subjects, overall scene, and atmosphere. Provide a clear and concise description in one or two sentences, suitable for a text-to-image AI system."""

            poster_prompt = f"""Analyze the provided images and extract key information to create a cinamtic movie poster style description. Format the output as follows:

Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters.

Here is an example of a prompt: 
Title: Display the title "Verdant Spirits" in elegant and ethereal text, placed centrally at the top of the poster.

Main Character: Depict a serene and enchantingly beautiful woman with an aura of nature, her face partly adorned and encased with vibrant green foliage and delicate floral arrangements. She exudes an ethereal and mystical presence.

Background: The background should feature a dreamlike enchanted forest with lush greenery, vibrant flowers, and an ethereal glow emanating from the foliage. The scene should feel magical and otherworldly, suggesting a hidden world within nature.

Supporting Characters: Add an enigmatic skeletal figure entwined with glowing, bioluminescent leaves and plants, subtly blending with the dark, verdant background. This figure should evoke a sense of ancient wisdom and mysterious energy.

Studio Ghibli Branding: Incorporate the Studio Ghibli logo at the bottom center of the poster to establish it as an official Ghibli film.

Tagline: Include a tagline that reads: "Where Nature's Secrets Come to Life" prominently on the poster.

Visual Style: Ensure the overall visual style is consistent with Studio Ghibli s signature look   rich, detailed backgrounds, and characters imbued with a touch of whimsy and mystery. The colors should be lush and inviting, with an emphasis on the enchanting and mystical aspects of nature."""

            if poster:
                base_prompt = poster_prompt
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = (
                    default_happy_prompt if happy_talk else default_simple_prompt
                )

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100,
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            # Add the override at the beginning of the prompt
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            messages = [
                {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
            ]

            # Process each image in the batch
            for img_tensor in images:
                pil_image = self.tensor_to_pil(img_tensor)
                base64_image = self.encode_image(pil_image)
                messages[0]["content"].append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                    }
                )

            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages
            )
            self.save_prompt(response.choices[0].message.content)
            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            print(f"Images tensor shape: {images.shape}")
            print(f"Images tensor type: {images.dtype}")
            return (f"Error occurred while processing the request: {str(e)}",)
        
class GeminiTextOnly:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=self.gemini_api_key)
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "gemini_model": (["gemini-1.5-pro-latest", "gemini-1.5-pro-exp-0801", "gemini-1.5-flash"],),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("output", "clip_l")
    FUNCTION = "process_text"
    CATEGORY = "dagthomas"

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def save_prompt(self, prompt):
        filename_text = "gemini_text_only_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def process_text(
        self,
        custom_prompt="",
        additive_prompt="",
        tag="",
        sex="other",
        pronouns="them, their",
        dynamic_prompt=False,
        words="100",
        gemini_model="gemini-1.5-pro-latest",
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Generate a response."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)

            response = model.generate_content(full_prompt)

            result = response.text

            self.save_prompt(result)

            return (
                result,
                self.extract_first_two_sentences(result),
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            return (error_message, error_message[:100])

class Gpt4VisionCloner:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
            },
            "optional": {
                "custom_prompt": ("STRING", {"multiline": True, "default": ""})
            },
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = (
        "formatted_output",
        "raw_json",
        "faded_image",
    )
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        # Determine orientation based on aspect ratio
        aspect_ratio = images[0].width / images[0].height
        vertical_stack = aspect_ratio > 1

        if vertical_stack:
            # Vertical stacking for wider images
            fade_height = int(images[0].height * (fade_percentage / 100))
            total_height = sum(img.height for img in images) - fade_height * (
                len(images) - 1
            )
            max_width = max(img.width for img in images)
            combined_image = Image.new("RGB", (max_width, total_height))

            y_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    y_offset = img.height - fade_height
                else:
                    for y in range(fade_height):
                        factor = y / fade_height
                        for x in range(max_width):
                            if x < images[i - 1].width and x < img.width:
                                pixel1 = images[i - 1].getpixel(
                                    (x, images[i - 1].height - fade_height + y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x, y_offset + y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((0, fade_height, img.width, img.height)),
                        (0, y_offset + fade_height),
                    )
                    y_offset += img.height - fade_height
        else:
            # Horizontal stacking for taller images
            fade_width = int(images[0].width * (fade_percentage / 100))
            total_width = sum(img.width for img in images) - fade_width * (
                len(images) - 1
            )
            max_height = max(img.height for img in images)
            combined_image = Image.new("RGB", (total_width, max_height))

            x_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    x_offset = img.width - fade_width
                else:
                    for x in range(fade_width):
                        factor = x / fade_width
                        for y in range(max_height):
                            if y < images[i - 1].height and y < img.height:
                                pixel1 = images[i - 1].getpixel(
                                    (images[i - 1].width - fade_width + x, y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x_offset + x, y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((fade_width, 0, img.width, img.height)),
                        (x_offset + fade_width, 0),
                    )
                    x_offset += img.width - fade_width

        return combined_image

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def save_prompt(self, prompt):
        filename_text = "vision_json_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def format_element(self, element):
        element_type = element.get("Type", "")
        description = element.get("Description", "").lower()
        attributes = element.get("Attributes", {})

        if element_type.lower() == "object":
            formatted = description
        else:
            formatted = f"{element_type} {description}"

        attr_details = []
        for key, value in attributes.items():
            if value and value.lower() not in ["n/a", "none", "None"]:
                attr_details.append(f"{key.lower()}: {value.lower()}")

        if attr_details:
            formatted += f" ({', '.join(attr_details)})"

        return formatted

    def extract_data(self, data):
        extracted = []
        extracted.append(data.get("Title", ""))
        extracted.append(data.get("Artistic Style", ""))

        color_scheme = data.get("Color Scheme", [])
        if color_scheme:
            extracted.append(
                f"palette ({', '.join(color.lower() for color in color_scheme)})"
            )

        for element in data.get("Elements", []):
            extracted.append(self.format_element(element))

        overall_scene = data.get("Overall Scene", {})
        extracted.append(overall_scene.get("Theme", ""))
        extracted.append(overall_scene.get("Setting", ""))
        extracted.append(overall_scene.get("Lighting", ""))

        return ", ".join(item for item in extracted if item)

    def analyze_images(self, images, fade_percentage=15.0, custom_prompt=""):
        try:
            default_prompt = """Analyze the provided image and generate a JSON object with the following structure:
{
"title": [A descriptive title for the image],
"color_scheme": [Array of dominant colors, including notes on significant contrasts or mood-setting choices],
"elements": [
{
"type": [Either "character" or "object"],
"description": [Brief description of the element],
"attributes": {
[Relevant attributes like clothing, accessories, position, location, category, etc.]
}
},
... [Additional elements]
],
"overall_scene": {
"theme": [Overall theme of the image],
"setting": [Where the scene takes place and how it contributes to the narrative],
"lighting": {
"type": [Type of lighting, e.g. natural, artificial, magical],
"direction": [Direction of the main light source],
"quality": [Quality of light, e.g. soft, harsh, diffused],
"effects": [Any special lighting effects or atmosphere created]
},
"mood": [The emotional tone or atmosphere conveyed],
"camera_angle": {
"perspective": [e.g. eye-level, low angle, high angle, bird's eye view],
"focus": [What the camera is focused on],
"depth_of_field": [Describe if all elements are in focus or if there's a specific focal point]
}
},

"artistic_choices": [Array of notable artistic decisions that contribute to the image's impact],
"text_elements": [
{
"content": [The text content],
"placement": [Description of where the text is placed in the image],
"style": [Description of the text style, font, color, etc.],
"purpose": [The role or purpose of the text in the overall composition]
},
... [Additional text elements]
]
}
ALWAYS blend concepts into one concept if there are multiple images. Ensure that all aspects of the image are thoroughly analyzed and accurately represented in the JSON output, including the camera angle, lighting details, and any significant distant objects or background elements. Provide the JSON output without any additional explanation or commentary."""

            final_prompt = custom_prompt if custom_prompt.strip() else default_prompt

            messages = [
                {"role": "user", "content": [{"type": "text", "text": final_prompt}]}
            ]

            # Handle single image or multiple images
            if len(images.shape) == 3:  # Single image
                pil_images = [self.tensor2pil(images)]
            else:  # Multiple images
                pil_images = [self.tensor2pil(img) for img in images]

            combined_image = self.fade_images(pil_images, fade_percentage)
            base64_image = self.encode_image(combined_image)

            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages
            )

            content = response.choices[0].message.content
            # print(content)

            # Check if the content is wrapped in Markdown code blocks
            if content.startswith("```json") and content.endswith("```"):
                # Remove the Markdown code block markers
                json_str = content[7:-3].strip()
            else:
                json_str = content

            # Parse the JSON
            data = json.loads(json_str)

            # Handle single image or multiple images
            if isinstance(data, list):
                results = [self.extract_data(image_data) for image_data in data]
                result = " | ".join(results)
            else:
                result = self.extract_data(data)

            self.save_prompt(result)
            faded_image_tensor = self.pil2tensor(combined_image)
            return (result, json.dumps(data, indent=2), faded_image_tensor)
        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, "{}", self.pil2tensor(error_image))

class GeminiCustomVision:
    def __init__(self):
        self.gemini_api_key = os.environ.get("GEMINI_API_KEY")
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")
        genai.configure(api_key=self.gemini_api_key)
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "gemini_model": (["gemini-1.5-pro-latest", "gemini-1.5-pro-exp-0801", "gemini-1.5-flash"],),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "analyze_images"
    CATEGORY = "dagthomas"

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        fade_width = int(images[0].width * (fade_percentage / 100))
        total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
        max_height = max(img.height for img in images)

        combined_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for i, img in enumerate(images):
            if i == 0:
                combined_image.paste(img, (0, 0))
                x_offset = img.width - fade_width
            else:
                for x in range(fade_width):
                    factor = x / fade_width
                    for y in range(max_height):
                        if y < images[i - 1].height and y < img.height:
                            pixel1 = images[i - 1].getpixel(
                                (images[i - 1].width - fade_width + x, y)
                            )
                            pixel2 = img.getpixel((x, y))
                            blended_pixel = tuple(
                                int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                for c in range(3)
                            )
                            combined_image.putpixel((x_offset + x, y), blended_pixel)

                combined_image.paste(
                    img.crop((fade_width, 0, img.width, img.height)),
                    (x_offset + fade_width, 0),
                )
                x_offset += img.width - fade_width

        return combined_image

    def save_prompt(self, prompt):
        filename_text = "gemini_custom_vision_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def analyze_images(
        self,
        images,
        custom_prompt="",
        additive_prompt="",
        tag="",
        sex="other",
        pronouns="them, their",
        dynamic_prompt=False,
        words="100",
        fade_percentage=15.0,
        gemini_model="gemini-1.5-pro-latest",
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)

            safety_settings = [
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE",
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE",
                },
            ]

            model = genai.GenerativeModel(gemini_model, safety_settings=safety_settings)


            response = model.generate_content(
                [full_prompt, combined_image],
            )

            result = response.text

            self.save_prompt(result)
            faded_image_tensor = self.pil2tensor(combined_image)

            return (
                result,
                self.extract_first_two_sentences(result),
                faded_image_tensor,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))
        
class Gpt4CustomVision:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "analyze_images"
    CATEGORY = CUSTOM_CATEGORY

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        # Determine orientation based on aspect ratio
        aspect_ratio = images[0].width / images[0].height
        vertical_stack = aspect_ratio > 1

        if vertical_stack:
            # Vertical stacking for wider images
            fade_height = int(images[0].height * (fade_percentage / 100))
            total_height = sum(img.height for img in images) - fade_height * (
                len(images) - 1
            )
            max_width = max(img.width for img in images)
            combined_image = Image.new("RGB", (max_width, total_height))

            y_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    y_offset = img.height - fade_height
                else:
                    for y in range(fade_height):
                        factor = y / fade_height
                        for x in range(max_width):
                            if x < images[i - 1].width and x < img.width:
                                pixel1 = images[i - 1].getpixel(
                                    (x, images[i - 1].height - fade_height + y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x, y_offset + y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((0, fade_height, img.width, img.height)),
                        (0, y_offset + fade_height),
                    )
                    y_offset += img.height - fade_height
        else:
            # Horizontal stacking for taller images
            fade_width = int(images[0].width * (fade_percentage / 100))
            total_width = sum(img.width for img in images) - fade_width * (
                len(images) - 1
            )
            max_height = max(img.height for img in images)
            combined_image = Image.new("RGB", (total_width, max_height))

            x_offset = 0
            for i, img in enumerate(images):
                if i == 0:
                    combined_image.paste(img, (0, 0))
                    x_offset = img.width - fade_width
                else:
                    for x in range(fade_width):
                        factor = x / fade_width
                        for y in range(max_height):
                            if y < images[i - 1].height and y < img.height:
                                pixel1 = images[i - 1].getpixel(
                                    (images[i - 1].width - fade_width + x, y)
                                )
                                pixel2 = img.getpixel((x, y))
                                blended_pixel = tuple(
                                    int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                    for c in range(3)
                                )
                                combined_image.putpixel(
                                    (x_offset + x, y), blended_pixel
                                )

                    combined_image.paste(
                        img.crop((fade_width, 0, img.width, img.height)),
                        (x_offset + fade_width, 0),
                    )
                    x_offset += img.width - fade_width

        return combined_image

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def save_prompt(self, prompt):
        filename_text = "custom_vision_json_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def analyze_images(
        self,
        images,
        custom_prompt="",
        additive_prompt="",
        tag="",
        sex="other",
        pronouns="them, their",
        dynamic_prompt=False,
        words="100",
        fade_percentage=15.0,
    ):
        try:
            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                if additive_prompt:
                    full_prompt = f"{additive_prompt} {custom_prompt}".strip()
                else:
                    full_prompt = (
                        custom_prompt if custom_prompt else "Analyze this image."
                    )

            messages = [
                {"role": "user", "content": [{"type": "text", "text": full_prompt}]}
            ]

            # Convert tensor images to PIL
            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            base64_image = self.encode_image(combined_image)

            messages[0]["content"].append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{base64_image}"},
                }
            )

            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages
            )

            try:
                self.save_prompt(response.choices[0].message.content)
            except Exception as e:
                print(f"Failed to save prompt: {e}")

            faded_image_tensor = self.pil2tensor(combined_image)

            return (
                response.choices[0].message.content,
                self.extract_first_two_sentences(response.choices[0].message.content),
                faded_image_tensor,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))


class RandomIntegerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "min_value": (
                    "INT",
                    {"default": 0, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
                "max_value": (
                    "INT",
                    {"default": 10, "min": -1000000000, "max": 1000000000, "step": 1},
                ),
                "seed": (
                    "INT",
                    {"default": -1, "min": -1, "max": 2**32 - 1, "step": 1},
                ),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "generate_random_int"
    CATEGORY = CUSTOM_CATEGORY

    def generate_random_int(self, min_value, max_value, seed):
        if min_value > max_value:
            min_value, max_value = max_value, min_value

        if seed != -1:
            random.seed(seed)

        return (random.randint(min_value, max_value),)

    @classmethod
    def IS_CHANGED(cls, min_value, max_value, seed):
        return seed != -1

    @classmethod
    def VALIDATE_INPUTS(cls, min_value, max_value, seed):
        if min_value > max_value:
            return "Minimum value should be less than or equal to maximum value."
        if seed < -1 or seed >= 2**32:
            return "Seed should be between -1 and 2^32 - 1."
        return True

    @classmethod
    def control_after_generate(cls, generated_value, min_value, max_value, seed):
        if generated_value < min_value or generated_value > max_value:
            return (max(min(generated_value, max_value), min_value),)
        return (generated_value,)


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
                "poster": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_model": ("STRING", {"default": "llama3.1:8b"}),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434/api/generate"},
                ),
                "custom_title": ("STRING", {"default": ""}),  # New custom title field
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),  # New override field
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def save_prompt(self, prompt):
        filename_text = "mini_" + prompt.split(",")[0].strip()
        filename_text = re.sub(r"[^\w\-_\. ]", "_", filename_text)
        filename_text = filename_text[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        with open(filename, "w") as file:
            file.write(prompt)

        print(f"Prompt saved to {filename}")

    def generate(
        self,
        input_text,
        happy_talk,
        compress,
        compression_level,
        poster,
        custom_base_prompt="",
        custom_model="llama3.1:8b",
        ollama_url="http://localhost:11434/api/generate",
        custom_title="",
        override="",
    ):
        try:
            default_happy_prompt = """Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:
If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style
Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus
Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 150 WORDS"""

            default_simple_prompt = """Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:
If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style
Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus
Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 200 WORDS"""  ##

            poster_prompt = f"""
Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Describe the image as a professional art critic. Create a cohesive, realistic scene in a single paragraph. Include:

If you find text in the image:
Quote exact text and output it and state it is a big title as: "[exact text]"
Describe placement
Suggest fitting font/style

Main subject details
Artistic style and theme
Setting and narrative contribution
Lighting characteristics
Color palette and emotional tone
Camera angle and focus

Merge image concepts if there is more than one.
Always blend the concepts, never talk about splits or parallel. 
Do not split or divide scenes, or talk about them differently - merge everything to one scene and one scene only.
Blend all elements into unified reality. Use image generation prompt language. No preamble, questions, or commentary.
CRITICAL: TRY TO OUTPUT ONLY IN 75 WORDS"""

            if poster:
                base_prompt = poster_prompt
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = (
                    default_happy_prompt if happy_talk else default_simple_prompt
                )

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100,
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            # Add the override at the beginning of the prompt if provided
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            prompt = f"{final_prompt}\nDescription: {input_text}"

            payload = {"model": custom_model, "prompt": prompt, "stream": False}

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            self.save_prompt(result)
            return (result,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)

class OllamaVisionNode:
    def __init__(self):
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)
        self.loaded_model = None

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "custom_model": ("STRING", {"default": "llava-llama3:latest"}),
                "ollama_url": (
                    "STRING",
                    {"default": "http://localhost:11434/api/generate"},
                ),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "IMAGE",
    )
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "analyze_images"
    CATEGORY = "dagthomas"

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        fade_width = int(images[0].width * (fade_percentage / 100))
        total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
        max_height = max(img.height for img in images)

        combined_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for i, img in enumerate(images):
            if i == 0:
                combined_image.paste(img, (0, 0))
                x_offset = img.width - fade_width
            else:
                for x in range(fade_width):
                    factor = x / fade_width
                    for y in range(max_height):
                        if y < images[i - 1].height and y < img.height:
                            pixel1 = images[i - 1].getpixel(
                                (images[i - 1].width - fade_width + x, y)
                            )
                            pixel2 = img.getpixel((x, y))
                            blended_pixel = tuple(
                                int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                for c in range(3)
                            )
                            combined_image.putpixel((x_offset + x, y), blended_pixel)

                combined_image.paste(
                    img.crop((fade_width, 0, img.width, img.height)),
                    (x_offset + fade_width, 0),
                )
                x_offset += img.width - fade_width

        return combined_image

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def save_prompt(self, prompt):
        filename_text = "ollama_vision_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def check_ollama_service(self, url):
            try:
                response = requests.get(url.replace("/api/generate", ""))
                return response.status_code == 200
            except requests.RequestException:
                return False
            
    def check_and_load_model(self, model_name, ollama_url):
        if self.loaded_model != model_name:
            load_url = ollama_url.replace("/api/generate", "/api/pull")
            try:
                response = requests.post(load_url, json={"name": model_name})
                if response.status_code == 200:
                    print(f"Successfully loaded model: {model_name}")
                    self.loaded_model = model_name
                else:
                    raise Exception(f"Failed to load model: {model_name}. Status code: {response.status_code}")
            except requests.RequestException as e:
                raise Exception(f"Error while loading model: {e}")

    def unload_model(self, ollama_url):
        if self.loaded_model:
            unload_url = ollama_url.replace("/api/generate", "/api/delete")
            try:
                response = requests.delete(unload_url, json={"name": self.loaded_model})
                if response.status_code == 200:
                    print(f"Successfully unloaded model: {self.loaded_model}")
                    self.loaded_model = None
                else:
                    print(f"Failed to unload model: {self.loaded_model}. Status code: {response.status_code}")
            except requests.RequestException as e:
                print(f"Error while unloading model: {e}")

    def analyze_images(
        self,
        images,
        custom_prompt="",
        additive_prompt="",
        tag="",
        sex="other",
        pronouns="them, their",
        dynamic_prompt=False,
        words="100",
        fade_percentage=15.0,
        custom_model="llava-llama3:latest",
        ollama_url="http://localhost:11434/api/generate",
        unload_after_use=True,
    ):
        try:
            if not self.check_ollama_service(ollama_url):
                raise Exception("Ollama service is not running. Please start it with 'ollama serve'.")
            
            self.check_and_load_model(custom_model, ollama_url)

            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            encoded_image = self.encode_image(combined_image)

            payload = {
                "model": custom_model,
                "prompt": full_prompt,
                "images": [encoded_image],
                "stream": False
            }

            response = requests.post(ollama_url, json=payload)
            response.raise_for_status()
            result = response.json()["response"]

            self.save_prompt(result)
            faded_image_tensor = self.pil2tensor(combined_image)
            if unload_after_use:
                self.unload_model(ollama_url)

            return (
                result,
                self.extract_first_two_sentences(result),
                faded_image_tensor,
            )

        except requests.exceptions.RequestException as e:
            print(f"Request error: {e}")
            if e.response is not None:
                print(f"Response status code: {e.response.status_code}")
                print(f"Response content: {e.response.text}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))


class PhiModelPipeline:
    def __init__(self):
        self.model = None
        self.processor = None

class PhiModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["Phi-3.5-vision-instruct"],),
                "image_crops": ([4, 16], {"default": 4}),
                "attention_mechanism": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'eager'})
            },
        }

    RETURN_TYPES = ("PHI_MODEL_PIPELINE",)
    RETURN_NAMES = ("phi_pipeline",)
    FUNCTION = "load_phi_model"
    CATEGORY = "LLM/Phi"

    def load_phi_model(self, model_version, image_crops, attention_mechanism):
        model_id = f"microsoft/{model_version}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
        
        phi_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            attn_implementation=attention_mechanism
        )
        phi_processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=image_crops
        )

        phi_pipeline = PhiModelPipeline()
        phi_pipeline.model = phi_model
        phi_pipeline.processor = phi_processor

        return (phi_pipeline,)

class PhiModelInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phi_pipeline": ("PHI_MODEL_PIPELINE",),
                "user_prompt": ("STRING", {"default": '', "multiline": True}),
                "input_images": ("IMAGE",),
                "generation_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_phi_inference"
    CATEGORY = "LLM/Phi"

    def run_phi_inference(self, phi_pipeline, user_prompt, input_images, generation_temperature, max_output_tokens):
        if phi_pipeline.model is None or phi_pipeline.processor is None:
            print("Model or processor not loaded. Attempting to reload...")
            try:
                model_id = "microsoft/phi-2"  # Adjust this if you're using a different model
                model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
                
                phi_pipeline.model = AutoModelForCausalLM.from_pretrained(
                    model_checkpoint, 
                    device_map="cuda", 
                    torch_dtype="auto", 
                    trust_remote_code=True
                )
                phi_pipeline.processor = AutoProcessor.from_pretrained(model_id, 
                    trust_remote_code=True
                )
                print("Model and processor reloaded successfully.")
            except Exception as e:
                print(f"Failed to reload model and processor: {str(e)}")
                return ("Error: Failed to load model and processor. Please run the Phi Model Loader node again.",)

        phi_model = phi_pipeline.model
        phi_processor = phi_pipeline.processor
        
        image_placeholders = ""
        processed_images = []
        for index, image in enumerate(input_images, 1):
            image_tensor = torch.unsqueeze(image, 0)
            pil_image = tensor2pil(image_tensor).convert('RGB')
            processed_images.append(pil_image)
            image_placeholders += f"<|image_{index}|>\n"
        
        messages = [
            {"role": "user", "content": user_prompt + image_placeholders},
        ]
        
        formatted_prompt = phi_processor.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        model_inputs = phi_processor(formatted_prompt, processed_images, return_tensors="pt").to("cuda:0") 
        
        do_sample = generation_temperature > 0
        generation_config = { 
            "max_new_tokens": max_output_tokens, 
            "do_sample": do_sample,
        }
        if do_sample:
            generation_config["temperature"] = generation_temperature
        
        generated_ids = phi_model.generate(
            **model_inputs, 
            eos_token_id=phi_processor.tokenizer.eos_token_id, 
            **generation_config
        )
        generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
        generated_text = phi_processor.batch_decode(generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False)[0]
        
        # Instead of unloading, we'll keep the model and processor loaded
        # This might use more memory but will prevent the need to reload frequently
        # If memory is a concern, you might want to implement a more sophisticated
        # caching mechanism or adjust this approach based on your specific needs
        
        return (generated_text,)
    
class PhiCustomModelInference:
    def __init__(self):
        self.prompts_dir = "./custom_nodes/comfyui_dagthomas/prompts"
        os.makedirs(self.prompts_dir, exist_ok=True)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phi_pipeline": ("PHI_MODEL_PIPELINE",),
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "generation_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "run_phi_custom_inference"
    CATEGORY = "LLM/Phi"

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        fade_width = int(images[0].width * (fade_percentage / 100))
        total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
        max_height = max(img.height for img in images)

        combined_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for i, img in enumerate(images):
            if i == 0:
                combined_image.paste(img, (0, 0))
                x_offset = img.width - fade_width
            else:
                for x in range(fade_width):
                    factor = x / fade_width
                    for y in range(max_height):
                        if y < images[i - 1].height and y < img.height:
                            pixel1 = images[i - 1].getpixel(
                                (images[i - 1].width - fade_width + x, y)
                            )
                            pixel2 = img.getpixel((x, y))
                            blended_pixel = tuple(
                                int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                for c in range(3)
                            )
                            combined_image.putpixel((x_offset + x, y), blended_pixel)

                combined_image.paste(
                    img.crop((fade_width, 0, img.width, img.height)),
                    (x_offset + fade_width, 0),
                )
                x_offset += img.width - fade_width

        return combined_image

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def save_prompt(self, prompt):
        filename_text = "phi_custom_vision_" + "".join(
            c if c.isalnum() or c in "-_" else "_" for c in prompt[:30]
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        try:
            with open(filename, "w", encoding="utf-8") as file:
                file.write(prompt)
            print(f"Prompt saved to {filename}")
        except Exception as e:
            print(f"Error saving prompt: {e}")

    def run_phi_custom_inference(
        self,
        phi_pipeline,
        images,
        custom_prompt,
        additive_prompt,
        dynamic_prompt,
        tag,
        sex,
        words,
        pronouns,
        fade_percentage,
        generation_temperature,
        max_output_tokens
    ):
        try:
            if phi_pipeline.model is None or phi_pipeline.processor is None:
                print("Model or processor not loaded. Attempting to reload...")
                try:
                    model_id = "microsoft/phi-2"  # Adjust this if you're using a different model
                    model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
                    
                    phi_pipeline.model = AutoModelForCausalLM.from_pretrained(
                        model_checkpoint, 
                        device_map="cuda", 
                        torch_dtype="auto", 
                        trust_remote_code=True
                    )
                    phi_pipeline.processor = AutoProcessor.from_pretrained(model_id, 
                        trust_remote_code=True
                    )
                    print("Model and processor reloaded successfully.")
                except Exception as e:
                    print(f"Failed to reload model and processor: {str(e)}")
                    return ("Error: Failed to load model and processor. Please run the Phi Model Loader node again.", "", None)

            phi_model = phi_pipeline.model
            phi_processor = phi_pipeline.processor

            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            
            image_placeholders = ""
            processed_images = []
            for index, image in enumerate(pil_images, 1):
                processed_images.append(image.convert('RGB'))
                image_placeholders += f"<|image_{index}|>\n"
            
            messages = [
                {"role": "user", "content": full_prompt + image_placeholders},
            ]
            
            formatted_prompt = phi_processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = phi_processor(formatted_prompt, processed_images, return_tensors="pt").to("cuda:0") 
            
            do_sample = generation_temperature > 0
            generation_config = { 
                "max_new_tokens": max_output_tokens, 
                "do_sample": do_sample,
            }
            if do_sample:
                generation_config["temperature"] = generation_temperature
            
            generated_ids = phi_model.generate(
                **model_inputs, 
                eos_token_id=phi_processor.tokenizer.eos_token_id, 
                **generation_config
            )
            generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
            generated_text = phi_processor.batch_decode(generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False)[0]

            self.save_prompt(generated_text)
            faded_image_tensor = self.pil2tensor(combined_image)

            return (
                generated_text,
                self.extract_first_two_sentences(generated_text),
                faded_image_tensor,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))   
               
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
                "poster": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "custom_base_prompt": ("STRING", {"multiline": True, "default": ""}),
                "custom_title": ("STRING", {"default": ""}),  # New custom title field
                "override": (
                    "STRING",
                    {"multiline": True, "default": ""},
                ),  # New override field
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate"
    CATEGORY = CUSTOM_CATEGORY

    def save_prompt(self, prompt):
        filename_text = "mini_" + prompt.split(",")[0].strip()
        filename_text = re.sub(r"[^\w\-_\. ]", "_", filename_text)
        filename_text = filename_text[:30]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{filename_text}_{timestamp}.txt"
        filename = os.path.join(self.prompts_dir, base_filename)

        with open(filename, "w") as file:
            file.write(prompt)

        print(f"Prompt saved to {filename}")

    def generate(
        self,
        input_text,
        happy_talk,
        compress,
        compression_level,
        poster,
        custom_base_prompt="",
        custom_title="",
        override="",
    ):
        try:
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image").Remove any mention of digital artwork or artwork style. Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Including details about the film grain, color grading, and any artifacts or characteristics specific film and photography"""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            poster_prompt = f"""Analyze the provided description and extract key information to create a movie poster style description. Format the output as follows:

Title: {"Use the title '" + custom_title + "'" if poster and custom_title else "A catchy, intriguing title that captures the essence of the scene"}, place the title in "".
Main character: Give a description of the main character.
Background: Describe the background in detail.
Supporting characters: Describe the supporting characters
Branding type: Describe the branding type
Tagline: Include a tagline that captures the essence of the movie.
Visual style: Ensure that the visual style fits the branding type and tagline.
You are allowed to make up film and branding names, and do them like 80's, 90's or modern movie posters.

Output should NEVER be markdown or include any formatting.
NEVER ADD ANY HGAPPY TALK IN THE PROMPT
Add how the title should look and state that its the main title.
Here is an example of a prompt, make the output like a movie poster: 
Title: Display the title "Verdant Spirits" in elegant and ethereal text, placed centrally at the top of the poster.

Main Character: Depict a serene and enchantingly beautiful woman with an aura of nature, her face partly adorned and encased with vibrant green foliage and delicate floral arrangements. She exudes an ethereal and mystical presence.

Background: The background should feature a dreamlike enchanted forest with lush greenery, vibrant flowers, and an ethereal glow emanating from the foliage. The scene should feel magical and otherworldly, suggesting a hidden world within nature.

Supporting Characters: Add an enigmatic skeletal figure entwined with glowing, bioluminescent leaves and plants, subtly blending with the dark, verdant background. This figure should evoke a sense of ancient wisdom and mysterious energy.

Studio Ghibli Branding: Incorporate the Studio Ghibli logo at the bottom center of the poster to establish it as an official Ghibli film.

Tagline: Include a tagline that reads: "Where Nature's Secrets Come to Life" prominently on the poster.

Visual Style: Ensure the overall visual style is consistent with Studio Ghibli s signature look   rich, detailed backgrounds, and characters imbued with a touch of whimsy and mystery. The colors should be lush and inviting, with an emphasis on the enchanting and mystical aspects of nature."""

            if poster:
                base_prompt = poster_prompt
            elif custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = (
                    default_happy_prompt if happy_talk else default_simple_prompt
                )

            if compress and not poster:
                compression_chars = {
                    "soft": 600 if happy_talk else 300,
                    "medium": 400 if happy_talk else 200,
                    "hard": 200 if happy_talk else 100,
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be concise while retaining key visual details. MAX OUTPUT SIZE no more than {char_limit} characters."

            # Add the override at the beginning of the prompt if provided
            final_prompt = f"{override}\n\n{base_prompt}" if override else base_prompt

            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": f"{final_prompt}\nDescription: {input_text}",
                    }
                ],
            )

            self.save_prompt(response.choices[0].message.content)
            return (response.choices[0].message.content,)
        except Exception as e:
            print(f"An error occurred: {e}")
            return (f"Error occurred while processing the request: {str(e)}",)


class PromptGenerator:
    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "STRING")
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
        cleaned_string = re.sub(r",\s*,", ",", input_string)
        return cleaned_string

    def process_string(self, replaced, seed):
        replaced = re.sub(r"\s*,\s*", ",", replaced)
        replaced = re.sub(r",+", ",", replaced)
        original = replaced

        # Find the indices for "BREAK_CLIPL"
        first_break_clipl_index = replaced.find("BREAK_CLIPL")
        second_break_clipl_index = replaced.find(
            "BREAK_CLIPL", first_break_clipl_index + len("BREAK_CLIPL")
        )

        # Extract the content between "BREAK_CLIPL" markers
        if first_break_clipl_index != -1 and second_break_clipl_index != -1:
            clip_content_l = replaced[
                first_break_clipl_index + len("BREAK_CLIPL") : second_break_clipl_index
            ]

            # Update the replaced string by removing the "BREAK_CLIPL" content
            replaced = replaced[:first_break_clipl_index].strip(", ") + replaced[
                second_break_clipl_index + len("BREAK_CLIPL") :
            ].strip(", ")

            clip_l = clip_content_l
        else:
            clip_l = ""

        # Find the indices for "BREAK_CLIPG"
        first_break_clipg_index = replaced.find("BREAK_CLIPG")
        second_break_clipg_index = replaced.find(
            "BREAK_CLIPG", first_break_clipg_index + len("BREAK_CLIPG")
        )

        # Extract the content between "BREAK_CLIPG" markers
        if first_break_clipg_index != -1 and second_break_clipg_index != -1:
            clip_content_g = replaced[
                first_break_clipg_index + len("BREAK_CLIPG") : second_break_clipg_index
            ]

            # Update the replaced string by removing the "BREAK_CLIPG" content
            replaced = replaced[:first_break_clipg_index].strip(", ") + replaced[
                second_break_clipg_index + len("BREAK_CLIPG") :
            ].strip(", ")

            clip_g = clip_content_g
        else:
            clip_g = ""

        t5xxl = replaced

        original = original.replace("BREAK_CLIPL", "").replace("BREAK_CLIPG", "")
        original = re.sub(r"\s*,\s*", ",", original)
        original = re.sub(r",+", ",", original)
        clip_l = re.sub(r"\s*,\s*", ",", clip_l)
        clip_l = re.sub(r",+", ",", clip_l)
        clip_g = re.sub(r"\s*,\s*", ",", clip_g)
        clip_g = re.sub(r",+", ",", clip_g)
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
        # print("")
        # print(f"clip_l: {clip_l}")
        # print("")
        # print(f"clip_g: {clip_g}")
        # print("")
        # print(f"t5xxl: {t5xxl}")
        # print("")

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
            if (
                kwargs.get("photography_style", "") != "disabled"
                and kwargs.get("default_tags", "") != "disabled"
                or subject != ""
            ):
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
                    selected_subject = (
                        self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS)
                        .replace("a ", "")
                        .replace("an ", "")
                    )
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
                    selected_subject = (
                        self.get_choice(kwargs.get("default_tags", ""), DEFAULT_TAGS)
                        .replace("a ", "")
                        .replace("an ", "")
                    )
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
        if (
            kwargs.get("clothing", "") != "disabled"
            and kwargs.get("clothing", "") != "random"
        ):
            components.append(", dressed in ")
            clothing = kwargs.get("clothing", "")
            components.append(clothing)
        elif kwargs.get("clothing", "") == "random":
            components.append(", dressed in ")
            clothing = self.get_choice(kwargs.get("clothing", ""), CLOTHING)
            components.append(clothing)

        if (
            kwargs.get("composition", "") != "disabled"
            and kwargs.get("composition", "") != "random"
        ):
            components.append(",")
            composition = kwargs.get("composition", "")
            components.append(composition)
        elif kwargs.get("composition", "") == "random":
            components.append(",")
            composition = self.get_choice(kwargs.get("composition", ""), COMPOSITION)
            components.append(composition)

        if kwargs.get("pose", "") != "disabled" and kwargs.get("pose", "") != "random":
            components.append(",")
            pose = kwargs.get("pose", "")
            components.append(pose)
        elif kwargs.get("pose", "") == "random":
            components.append(",")
            pose = self.get_choice(kwargs.get("pose", ""), POSE)
            components.append(pose)
        components.append("BREAK_CLIPG")
        if (
            kwargs.get("background", "") != "disabled"
            and kwargs.get("background", "") != "random"
        ):
            components.append(",")
            background = kwargs.get("background", "")
            components.append(background)
        elif kwargs.get("background", "") == "random":
            components.append(",")
            background = self.get_choice(kwargs.get("background", ""), BACKGROUND)
            components.append(background)

        if (
            kwargs.get("place", "") != "disabled"
            and kwargs.get("place", "") != "random"
        ):
            components.append(",")
            place = kwargs.get("place", "")
            components.append(place)
        elif kwargs.get("place", "") == "random":
            components.append(",")
            place = self.get_choice(kwargs.get("place", ""), PLACE)
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
                if (
                    photo_type_choice
                    and photo_type_choice != "random"
                    and photo_type_choice != "disabled"
                ):
                    random_value = round(self.rng.uniform(1.1, 1.5), 1)
                    components.append(f", ({photo_type_choice}:{random_value}), ")

            params = [
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
                components.append(
                    f"by {self.get_choice(kwargs.get('artist', ''), ARTIST)}"
                )
        components.append("BREAK_CLIPL")

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        print(f"PromptGenerator Seed  : {seed}")
        replaced = prompt.replace("of as", "of")
        replaced = self.clean_consecutive_commas(replaced)

        return self.process_string(replaced, seed)


class APNextNode:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "random")
    FUNCTION = "process"
    CATEGORY = CUSTOM_CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        inputs = {
            "required": {
                "prompt": ("STRING", {"multiline": True, "lines": 4}),
                "separator": ("STRING", {"default": ","}),
            },
            "optional": {
                "string": ("STRING", {"default": "", "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "attributes": ("BOOLEAN", {"default": False}),
            },
        }
        category_path = os.path.join(
            os.path.dirname(__file__), "data", "next", cls._subcategory.lower()
        )
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.endswith(".json"):
                    field_name = file[:-5]
                    file_path = os.path.join(category_path, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:
                                data = json.loads(content)
                                options = (
                                    data.get("items", [])
                                    if isinstance(data, dict)
                                    else data
                                )
                                inputs["optional"][field_name] = (
                                    ["None", "Random", "Multiple Random"] + options,
                                    {"default": "None"},
                                )
                            else:
                                print(f"Warning: Empty file {file}")
                                inputs["optional"][field_name] = (
                                    ["None", "Random", "Multiple Random"],
                                    {"default": "None"},
                                )
                    except json.JSONDecodeError as e:
                        print(f"Error decoding JSON in file {file}: {e}")
                        inputs["optional"][field_name] = (
                            ["None", "Random", "Multiple Random"],
                            {"default": "None"},
                        )
                    except Exception as e:
                        print(f"Error processing file {file}: {e}")
                        inputs["optional"][field_name] = (
                            ["None", "Random", "Multiple Random"],
                            {"default": "None"},
                        )

        return inputs

    CATEGORY = CUSTOM_CATEGORY

    def __init__(self):
        self.data = self.load_json_data()

    def load_json_data(self):
        data = {}
        category_path = os.path.join(
            os.path.dirname(__file__), "data", "next", self._subcategory.lower()
        )
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                if file.endswith(".json"):
                    file_path = os.path.join(category_path, file)
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                        if isinstance(json_data, dict):
                            data[file[:-5]] = {
                                "items": json_data.get("items", []),
                                "preprompt": json_data.get("preprompt", ""),
                                "separator": json_data.get("separator", ", "),
                                "endprompt": json_data.get("endprompt", ""),
                                "attributes": json_data.get("attributes", {}),
                            }
                        else:
                            data[file[:-5]] = {
                                "items": json_data,
                                "preprompt": "",
                                "separator": ", ",
                                "endprompt": "",
                                "attributes": {},
                            }
        return data

    def process(self, prompt, separator, string="", attributes=False, seed=0, **kwargs):
        random.seed(seed)
        prompt_additions = []
        random_additions = []

        for field, value in kwargs.items():
            if field in self.data:
                field_data = self.data[field]
                items = field_data["items"]

                # For the prompt output
                if value == "None":
                    prompt_items = []
                elif value == "Random":
                    prompt_items = [random.choice(items)]
                elif value == "Multiple Random":
                    count = random.randint(1, 3)
                    prompt_items = random.sample(items, min(count, len(items)))
                else:
                    prompt_items = [value]

                # For the random output
                random_choice = random.choice(["None", "Random", "Multiple Random"])
                if random_choice == "None":
                    random_items = []
                elif random_choice == "Random":
                    random_items = [random.choice(items)]
                else:  # Multiple Random
                    count = random.randint(1, 3)
                    random_items = random.sample(items, min(count, len(items)))

                self.format_and_add_items(
                    prompt_items, field_data, attributes, prompt_additions
                )
                self.format_and_add_items(
                    random_items, field_data, attributes, random_additions
                )

        if string:
            modified_prompt = f"{string} {prompt}"
        else:
            modified_prompt = prompt

        if prompt_additions:
            modified_prompt = f"{modified_prompt} {' '.join(prompt_additions)}"

        if separator:
            modified_prompt = f"{modified_prompt}{separator}"

        # Construct random output, including 'string' if it exists
        random_output = ""
        if string:
            random_output += string
        if random_additions:
            if random_output:
                random_output += " "
            random_output += " ".join(random_additions)
        if random_output and separator:
            random_output += separator

        return (modified_prompt, random_output)

    def format_and_add_items(self, selected_items, field_data, attributes, additions):
        if selected_items:
            preprompt = str(field_data["preprompt"]).strip()
            field_separator = f" {str(field_data['separator']).strip()} "
            endprompt = str(field_data["endprompt"]).strip()

            formatted_items = []
            for item in selected_items:
                item_str = str(item)
                if attributes and item_str in field_data["attributes"]:
                    item_attributes = field_data["attributes"].get(item_str, [])
                    if item_attributes:
                        selected_attributes = random.sample(
                            item_attributes, min(3, len(item_attributes))
                        )
                        formatted_items.append(
                            f"{item_str} ({', '.join(map(str, selected_attributes))})"
                        )
                    else:
                        formatted_items.append(item_str)
                else:
                    formatted_items.append(item_str)

            formatted_values = field_separator.join(formatted_items)

            formatted_addition = []
            if preprompt:
                formatted_addition.append(preprompt)
            formatted_addition.append(formatted_values)
            if endprompt:
                formatted_addition.append(endprompt)

            formatted_output = " ".join(formatted_addition).strip()
            additions.append(formatted_output)


class CustomPromptLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt_file": (s.get_prompt_files(),),
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "load_prompt"
    CATEGORY = CUSTOM_CATEGORY

    @staticmethod
    def get_prompt_files():
        return [f for f in os.listdir(prompt_dir) if f.endswith(".txt")]

    @classmethod
    def IS_CHANGED(s, prompt_file):
        return float("nan")  # This ensures the widget is always refreshed

    def load_prompt(self, prompt_file):
        file_path = os.path.join(prompt_dir, prompt_file)

        # Detect the file encoding
        with open(file_path, "rb") as file:
            raw_data = file.read()
            detected = chardet.detect(raw_data)
            encoding = detected["encoding"]

        # Read the file with the detected encoding
        try:
            with open(file_path, "r", encoding=encoding) as file:
                content = file.read()
        except UnicodeDecodeError:
            # If the detected encoding fails, try UTF-8 as a fallback
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    content = file.read()
            except UnicodeDecodeError:
                # If UTF-8 also fails, try ANSI (Windows-1252) as a last resort
                with open(file_path, "r", encoding="windows-1252") as file:
                    content = file.read()

        return (content,)

class FileReaderNode:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "file_path": (
                    "STRING",
                    {"default": "./custom_nodes/comfyui_dagthomas/concat/output.json"},
                ),
                "amount": ("INT", {"default": 10, "min": 1, "max": 100}),
                "custom_tag": ("STRING", {"default": ""}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF})
            },
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_prompt"
    CATEGORY = CUSTOM_CATEGORY

    def generate_prompt(
        self, file_path: str, amount: int, custom_tag: str, seed: int = 0
    ) -> tuple:
        try:
            # Set the random seed if provided
            if seed != 0:
                random.seed(seed)

            # Step 1: Load JSON data from the file with UTF-8 encoding
            with open(file_path, "r", encoding="utf-8") as file:
                json_list = json.load(file)

            # Step 2: Randomly select the specified number of elements from the list
            random_values = random.sample(json_list, min(amount, len(json_list)))

            # Step 3: Join the selected elements into a single string separated by commas
            result_string = ", ".join(random_values)

            # Step 4: Add the custom tag if provided
            if custom_tag:
                result_string = f"{custom_tag}, {result_string}"

            return (result_string,)

        except Exception as e:
            return (f"Error: {str(e)}",)


# This line is required for ComfyUI to recognize and load the node
NODE_CLASS_MAPPINGS = {"FileReaderNode": FileReaderNode}

NODE_CLASS_MAPPINGS = {
    "OllamaVisionNode": OllamaVisionNode,
    "FileReaderNode": FileReaderNode,
    "APNLatent": APNLatent,
    "CustomPromptLoader": CustomPromptLoader,
    "DynamicStringCombinerNode": DynamicStringCombinerNode,
    "SentenceMixerNode": SentenceMixerNode,
    "RandomIntegerNode": RandomIntegerNode,
    "OllamaNode": OllamaNode,
    "FlexibleStringMergerNode": FlexibleStringMergerNode,
    "StringMergerNode": StringMergerNode,
    "Gpt4CustomVision": Gpt4CustomVision,
    "GPT4VisionNode": GPT4VisionNode,
    "Gpt4VisionCloner": Gpt4VisionCloner,
    "GPT4MiniNode": GPT4MiniNode,
    "PromptGenerator": PromptGenerator,
    "PGSD3LatentGenerator": PGSD3LatentGenerator,
    "PhiModelLoader": PhiModelLoader,
    "PhiModelInference": PhiModelInference,
    "PhiCustomModelInference": PhiCustomModelInference,
    "GeminiCustomVision": GeminiCustomVision,
    "GeminiTextOnly": GeminiTextOnly,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "OllamaVisionNode": "APNext OllamaVision",
    "FileReaderNode": "APNext Local random prompt",
    "APNLatent": "APNext Latent Generator",
    "CustomPromptLoader": "APNext Custom Prompts",
    "DynamicStringCombinerNode": "APNext Dynamic String Combiner",
    "SentenceMixerNode": "APNext Sentence Mixer",
    "RandomIntegerNode": "APNext Random Integer Generator",
    "GPT4MiniNode": "APNext GPT-4o-mini generator",
    "PromptGenerator": "Auto Prompter",
    "PGSD3LatentGenerator": "APNext PGSD3LatentGenerator",
    "Gpt4CustomVision": "APNext Gpt4CustomVision",
    "GPT4VisionNode": "APNext GPT4VisionNode",
    "Gpt4VisionCloner": "APNext Gpt4VisionCloner",
    "StringMergerNode": "APNext String Merger",
    "FlexibleStringMergerNode": "APNext Flexible String Merger",
    "OllamaNode": "APNext OllamaNode",
    "PhiModelLoader": "APNext Phi Model Loader",
    "PhiModelInference": "APNext Phi Model Inference",
    "PhiCustomModelInference": "APNext Phi Custom Model Inference",
    "GeminiCustomVision": "APNext Gemini Custom Vision",
    "GeminiTextOnly": "APNext Gemini Text Only",
}

categories = [
    d for d in os.listdir(next_dir) if os.path.isdir(os.path.join(next_dir, d))
]

for category in categories:
    class_name = (
        f"{''.join(word.capitalize() for word in category.split('_'))}PromptNode"
    )
    new_class = type(
        class_name,
        (APNextNode,),
        {
            "CATEGORY": CUSTOM_CATEGORY,  # Set the CATEGORY to CUSTOM_CATEGORY
            "_subcategory": category,  # Add a _subcategory attribute
        },
    )
    globals()[class_name] = new_class
    NODE_CLASS_MAPPINGS[class_name] = new_class
    NODE_DISPLAY_NAME_MAPPINGS[class_name] = (
        f"APNext {category.replace('_', ' ').title()}"
    )
