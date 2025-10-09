# APNextNode Node

import os
import json
import random
from ...utils.constants import CUSTOM_CATEGORY


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
        
        # Check if this is a dynamically created subclass with _subcategory
        if not hasattr(cls, '_subcategory'):
            # This is the base APNextNode class, return basic inputs
            return inputs
            
        # Get the main package directory (go up 3 levels: apnext_nodes.py -> prompt_generators -> nodes -> comfyui_dagthomas)
        package_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        category_path = os.path.join(
            package_dir, "data", "next", cls._subcategory.lower()
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

