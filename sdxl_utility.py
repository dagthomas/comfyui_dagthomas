# Converting the prompt generation script into a ComfyUI plugin structure
import random
import json
import comfy.sd
import comfy.model_management
import nodes
import torch
import os
import requests
from openai import OpenAI

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

    CATEGORY = "latent/sd3"

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

class GPT4MiniNode:
    def __init__(self):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

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
    CATEGORY = "text"

    def generate(self, input_text, happy_talk, compress, compression_level, custom_base_prompt=""):
        try:
            default_happy_prompt = """Create a detailed visually descriptive caption of this description, which will be used as a prompt for a text to image AI system (caption only, no instructions like "create an image"). Give detailed visual descriptions of the character(s), including ethnicity, skin tone, expression etc. Imagine using keywords for a still for someone who has aphantasia. Describe the image style, e.g. any photographic or art styles / techniques utilized. Make sure to fully describe all aspects of the cinematography, with abundant technical details and visual descriptions. If there is more than one image, combine the elements and characters from all of the images creatively into a single cohesive composition with a single background, inventing an interaction between the characters. Be creative in combining the characters into a single cohesive scene. Focus on two primary characters (or one) and describe an interesting interaction between them, such as a hug, a kiss, a fight, giving an object, an emotional reaction / interaction. If there is more than one background in the images, pick the most appropriate one. Your output is only the caption itself, no comments or extra formatting. The caption is in a single long paragraph. If you feel the images are inappropriate, invent a new scene / characters inspired by these. Additionally, incorporate a specific movie director's visual style (e.g. Wes Anderson, Christopher Nolan, Quentin Tarantino) and describe the lighting setup in detail, including the type, color, and placement of light sources to create the desired mood and atmosphere. Always frame the scene as a screen grab from a 35mm film still, including details about the film grain, color grading, and any artifacts or characteristics specific to 35mm film photography."""

            default_simple_prompt = """Create a brief, straightforward caption for this description, suitable for a text-to-image AI system. Focus on the main elements, key characters, and overall scene without elaborate details. Provide a clear and concise description in one or two sentences."""

            if custom_base_prompt.strip():
                base_prompt = custom_base_prompt
            else:
                base_prompt = default_happy_prompt if happy_talk else default_simple_prompt

            if compress and happy_talk:
                compression_chars = {
                    "soft": 400,
                    "medium": 300,
                    "hard": 200
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Compress the output to be more concise while retaining key visual details. The response should be no more than {char_limit} characters."
            elif compress and not happy_talk:
                compression_chars = {
                    "soft": 100,
                    "medium": 75,
                    "hard": 50
                }
                char_limit = compression_chars[compression_level]
                base_prompt += f" Limit the response to no more than {char_limit} characters."

            stream = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": f"{base_prompt}\nDescription: {input_text}"}],
                stream=True,
            )
            
            response_content = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    response_content += chunk.choices[0].delta.content
            
            return (response_content,)
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
    CATEGORY = "PromptGenerator"

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
        "GPT4MiniNode": GPT4MiniNode,
    "PromptGenerator": PromptGenerator,
    "PGSD3LatentGenerator": PGSD3LatentGenerator,
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GPT4MiniNode": "GPT-4o-mini morbuto Generator",
    "PromptGenerator": "Auto Prompter",
    "PGSD3LatentGenerator": "PGSD3LatentGenerator"
}
