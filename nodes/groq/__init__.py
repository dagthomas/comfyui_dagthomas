# Groq Nodes

from .text_node import GroqTextNode
from .vision_node import GroqVisionNode

NODE_CLASS_MAPPINGS = {
    "GroqTextNode": GroqTextNode,
    "GroqVisionNode": GroqVisionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GroqTextNode": "APNext Groq Text Generator",
    "GroqVisionNode": "APNext Groq Vision Analyzer",
}

