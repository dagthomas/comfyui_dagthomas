# GptVisionCloner Node

from ...utils.constants import CUSTOM_CATEGORY
from ...utils.constants import gpt_models
from ...utils.image_utils import tensor2pil, pil2tensor
from PIL import Image
from openai import OpenAI
import base64
import io
import json
import numpy as np
import os
import torch
import httpx


class GptVisionCloner:
    def __init__(self):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")

        # Create a compatible httpx client to avoid version conflicts
        try:
            http_client = httpx.Client(timeout=60.0)
        except TypeError:
            # Fallback for older httpx versions that don't support certain parameters
            http_client = httpx.Client()

        self.client = OpenAI(api_key=api_key, http_client=http_client)

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "gpt_model": (gpt_models,),
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

    def analyze_images(self, images, fade_percentage=15.0, gpt_model="gpt-5", custom_prompt=""):
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

            print(f"🔄 GPT Vision Cloner: Sending request to {gpt_model}...")

            # Timeout is handled by the httpx client
            response = self.client.chat.completions.create(
                model=gpt_model,
                messages=messages,
                max_tokens=4096,
                temperature=0.7
            )

            print(f"✅ GPT Vision Cloner: Received response from {gpt_model}")
            content = response.choices[0].message.content

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

            faded_image_tensor = self.pil2tensor(combined_image)
            return (result, json.dumps(data, indent=2), faded_image_tensor)
        except TimeoutError as e:
            print(f"❌ GPT Vision Cloner: Request timed out after 60 seconds")
            error_message = "Request timed out. OpenAI API took too long to respond."
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, "{}", self.pil2tensor(error_image))
        except ValueError as e:
            print(f"❌ GPT Vision Cloner: {str(e)}")
            error_message = f"Configuration error: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, "{}", self.pil2tensor(error_image))
        except json.JSONDecodeError as e:
            print(f"❌ GPT Vision Cloner: Failed to parse JSON response: {e}")
            error_message = f"Invalid JSON response from API: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, "{}", self.pil2tensor(error_image))
        except Exception as e:
            import traceback
            print(f"❌ GPT Vision Cloner: Unexpected error: {e}")
            print(traceback.format_exc())
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, "{}", self.pil2tensor(error_image))
