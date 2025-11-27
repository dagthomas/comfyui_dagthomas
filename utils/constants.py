# Constants and shared data for all nodes

import os
import json
import codecs

# Category for all nodes
CUSTOM_CATEGORY = "comfyui_dagthomas"

def load_json_file(file_name):
    """Load data from a JSON file in the data directory"""
    file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", file_name)
    with open(file_path, "r") as file:
        return json.load(file)

def load_all_json_files(base_path):
    """Load all JSON files from a directory recursively"""
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

# Load base directory paths
base_dir = os.path.dirname(os.path.dirname(__file__))
next_dir = os.path.join(base_dir, "data", "next")
prompt_dir = os.path.join(base_dir, "data", "custom_prompts")

# Load all JSON data
all_data = load_all_json_files(next_dir)

# Load individual JSON files
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

# Load model configurations
models_file = os.path.join(base_dir, "data", "gemini_models.json")
with open(models_file, 'r') as f:
    models_data = json.load(f)
    gemini_models = models_data.get('models', [])

gpt_models_file = os.path.join(base_dir, "data", "gpt_models.json")
with open(gpt_models_file, 'r') as f:
    gpt_models_data = json.load(f)
    gpt_models = gpt_models_data.get('models', [])

grok_models_file = os.path.join(base_dir, "data", "grok_models.json")
with open(grok_models_file, 'r') as f:
    grok_models_data = json.load(f)
    grok_models = grok_models_data.get('models', [])

claude_models_file = os.path.join(base_dir, "data", "claude_models.json")
with open(claude_models_file, 'r') as f:
    claude_models_data = json.load(f)
    claude_models = claude_models_data.get('models', [])

qwenvl_models_file = os.path.join(base_dir, "data", "qwenvl_models.json")
with open(qwenvl_models_file, 'r') as f:
    qwenvl_models_data = json.load(f)
    qwenvl_models = qwenvl_models_data.get('models', [])

def load_groq_models_from_file():
    """Load Groq models from JSON file."""
    try:
        groq_models_file = os.path.join(base_dir, "data", "groq_models.json")
        with open(groq_models_file, 'r') as f:
            data = json.load(f)
            
            # Support both old format (models array) and new format (text_models/vision_models)
            if 'text_models' in data:
                text_models = data.get('text_models', [])
                vision_models = data.get('vision_models', [])
                print(f"Loaded {len(text_models)} text models and {len(vision_models)} vision models from JSON file")
                return text_models, vision_models
            else:
                # Old format compatibility
                models = data.get('models', [])
                print(f"Loaded {len(models)} Groq models from JSON file")
                return models, []
    except Exception as e:
        print(f"Warning: Could not load Groq models from file: {e}")
        # Return minimal defaults
        default_text = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "groq/compound"]
        default_vision = ["llama-4-scout-17b-16e-instruct", "meta-llama/llama-4-scout-17b-16e-instruct"]
        return default_text, default_vision

def get_groq_models_from_api():
    """
    Fetch available Groq models dynamically from API.
    Returns (text_models, vision_models) tuple.
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return None, None
    
    try:
        import requests
        response = requests.get(
            "https://api.groq.com/openai/v1/models",
            headers={
                "Authorization": f"Bearer {groq_api_key}",
                "Content-Type": "application/json"
            },
            timeout=5
        )
        if response.status_code == 200:
            models_data = response.json()
            all_models = [model['id'] for model in models_data.get('data', [])]
            
            # Filter out non-text models (audio, tts, guards, etc.)
            text_models = [m for m in all_models if not any(x in m.lower() for x in ['whisper', 'tts', 'guard'])]
            # Vision models have 'vision' in the name or certain model types
            vision_models = [m for m in all_models if 'vision' in m.lower()]

            if text_models:
                print(f"Fetched {len(text_models)} text models and {len(vision_models)} vision models from Groq API")
                return sorted(text_models), sorted(vision_models)
    except Exception as e:
        print(f"Warning: Could not fetch Groq models from API: {e}")
    
    return None, None

# Load Groq models - try API first, then fall back to JSON file
groq_text_models, groq_vision_models = get_groq_models_from_api()
if groq_text_models is None:
    groq_text_models, groq_vision_models = load_groq_models_from_file()

# For backward compatibility, provide 'groq_models' as the text models list
groq_models = groq_text_models

# Legacy constants that were in sdxl_utility.py (now moved to utils)
def tensor2pil(t_image):
    """Legacy tensor to PIL conversion function"""
    import torch
    import numpy as np
    from PIL import Image
    return Image.fromarray(np.clip(255.0 * t_image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Cinematic terms for prompt enhancement
CINEMATIC_TERMS = {
    "visual_style": [
        "cinematic", "dramatic", "moody", "atmospheric", "ethereal", "surreal",
        "photorealistic", "hyperrealistic", "stylized", "abstract", "minimalist",
        "maximalist", "vintage", "retro", "futuristic", "cyberpunk", "steampunk"
    ],
    "lighting_type": [
        "soft lighting", "hard lighting", "dramatic lighting", "natural lighting",
        "studio lighting", "golden hour", "blue hour", "twilight", "dawn",
        "dusk", "overcast", "backlit", "rim lighting", "key lighting"
    ],
    "light_source": [
        "sunlight", "moonlight", "candlelight", "firelight", "neon lights",
        "street lamps", "window light", "overhead lighting", "side lighting",
        "bottom lighting", "practical lights", "ambient lighting"
    ],
    "camera_angle": [
        "low angle", "high angle", "eye level", "bird's eye view", "worm's eye view",
        "dutch angle", "overhead shot", "ground level", "three-quarter view"
    ],
    "shot_size": [
        "extreme wide shot", "wide shot", "medium wide shot", "medium shot",
        "medium close-up", "close-up", "extreme close-up", "establishing shot"
    ],
    "lens_type": [
        "wide-angle lens", "telephoto lens", "macro lens", "fisheye lens",
        "portrait lens", "zoom lens", "prime lens", "tilt-shift lens"
    ],
    "color_tone": [
        "warm tones", "cool tones", "desaturated", "vibrant", "monochromatic",
        "high contrast", "low contrast", "sepia", "black and white", "duotone"
    ],
    "camera_movement": [
        "static shot", "pan left", "pan right", "tilt up", "tilt down",
        "dolly in", "dolly out", "tracking shot", "crane shot", "handheld"
    ],
    "time_of_day": [
        "sunrise", "morning", "midday", "afternoon", "sunset", "night",
        "midnight", "pre-dawn", "twilight", "golden hour", "blue hour"
    ],
    "visual_effects": [
        "depth of field", "bokeh", "lens flare", "motion blur", "film grain",
        "vignette", "chromatic aberration", "light rays", "fog", "mist"
    ],
    "composition": [
        "rule of thirds", "centered composition", "symmetrical", "asymmetrical",
        "leading lines", "framing", "negative space", "foreground focus"
    ],
    "motion": [
        "static pose", "walking", "running", "jumping", "dancing", "flowing movement",
        "dramatic gesture", "subtle movement", "frozen action", "dynamic pose"
    ],
    "character_emotion": [
        "confident", "mysterious", "contemplative", "joyful", "melancholic",
        "determined", "serene", "intense", "playful", "stoic", "passionate"
    ]
}
