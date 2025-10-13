# Grok (xAI) Nodes

from .text_node import GrokTextNode
from .vision_node import GrokVisionNode

NODE_CLASS_MAPPINGS = {
    "GrokTextNode": GrokTextNode,
    "GrokVisionNode": GrokVisionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GrokTextNode": "APNext Grok Text Generator",
    "GrokVisionNode": "APNext Grok Vision Analyzer",
}
