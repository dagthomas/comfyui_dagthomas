# Converting the prompt generation script into a ComfyUI plugin structure
import random
import json

import os


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


class PromptGenerator:
    RETURN_TYPES = (
        "STRING",
        "INT",
    )
    RETURN_NAMES = (
        "prompt",
        "seed",
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
                    {"default": "random"},
                ),
                "photo_type": (
                    ["disabled"] + ["random"] + PHOTO_TYPE,
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

    def generate_prompt(self, **kwargs):
        seed = kwargs.get("seed", 0)
        if seed is not None:
            self.rng = random.Random(seed)
        components = []
        is_photographer = kwargs.get("artform", "").lower() == "photography" or (
            kwargs.get("artform", "").lower() == "random"
            and self.rng.choice([True, False])
        )

        if is_photographer:
            selected_photo_style = self.get_choice(
                kwargs.get("photography_styles", ""), PHOTOGRAPHY_STYLES
            )
            components.append(selected_photo_style)
            components.append(" of")
        custom = kwargs.get("custom", "")
        subject = kwargs.get("subject", "")
        default_tags = kwargs.get(
            "default_tags", "random"
        )  # default to "random" if not specified
        if custom:
            components.append(custom)
        if not subject:
            if default_tags == "random":
                selected_subject = self.get_choice(
                    kwargs.get("default_tags", ""), DEFAULT_TAGS
                )
                components.append(selected_subject)
            elif default_tags == "disabled":
                pass
            else:
                components.append(default_tags)
        else:
            components.append(subject)

        params = [
            ("roles", ROLES),
            ("hairstyles", HAIRSTYLES),
            ("additional_details", ADDITIONAL_DETAILS),
            ("place", PLACE),
            ("clothing", CLOTHING),
            ("composition", COMPOSITION),
            ("pose", POSE),
            ("background", BACKGROUND),
        ]
        for param in params:
            components.append(self.get_choice(kwargs.get(param[0], ""), param[1]))
        for i in reversed(range(len(components))):
            if components[i] in PLACE:
                components[i] += ","
                break
        lighting = kwargs.get("lighting", "").lower()
        if lighting == "random":
            selected_lighting = ", ".join(
                self.rng.sample(LIGHTING, self.rng.randint(2, 5))
            )
            components.append(selected_lighting)
        elif lighting == "disabled":
            pass
        else:
            components.append(lighting)
        if is_photographer:
            photo_type_choice = self.get_choice(
                kwargs.get("photo_type", ""), PHOTO_TYPE
            )
            # If a specific photo_type is chosen, use it directly in the prompt
            if photo_type_choice and photo_type_choice != "random":
                random_value = round(self.rng.uniform(1.1, 1.5), 1)
                components.append(f", ({photo_type_choice}:{random_value}), ")
            else:
                random_type = self.rng.choice(PHOTO_TYPE)
                components.append(f"{random_type}, ")
                random_type_float = round(random.uniform(1, 2), 1)
                formatted_type_value = f"({random_type}:{random_type_float})"
                random_framing = random.choice(PHOTO_FRAMING)
                random_framing_float = round(random.uniform(1, 2), 1)
                formatted_framing_value = f"({random_framing}:{random_framing_float})"

                components.append(f"{formatted_type_value} {formatted_framing_value} ")

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
            components[-2] = f"shot on {components[-2]}"
            components[-1] = f"photo by {components[-1]}"
        else:
            digital_artform_choice = self.get_choice(
                kwargs.get("digital_artform", ""), DIGITAL_ARTFORM
            )
            if digital_artform_choice:
                components.append(f"{digital_artform_choice}")
            components.append(f"by {self.get_choice(kwargs.get('artist', ''), ARTIST)}")

        prompt = " ".join(components)
        prompt = re.sub(" +", " ", prompt)
        print(f"PromptGenerator Seed  : {seed}")

        replaced = prompt.replace("of as", "of")
        replaced = replaced.replace(" , ", ", ")
        replaced = replaced.replace(". ", ", ")
        replaced = replaced.replace(";", ", ")
        print(f"PromptGenerator String: {replaced}")
        return (
            replaced,
            seed,
        )


NODE_CLASS_MAPPINGS = {
    "PromptGenerator": PromptGenerator,
    # "CSVPromptGenerator": CSVPromptGenerator,
    # "CSL": CommaSeparatedList,
}

# Human readable names for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "PromptGenerator": "SDXL Auto Prompter",
    "CSL": "Comma Separated List",
}
