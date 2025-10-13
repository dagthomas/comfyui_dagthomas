# Claude (Anthropic) Nodes

from .text_node import ClaudeTextNode
from .vision_node import ClaudeVisionNode

NODE_CLASS_MAPPINGS = {
    "ClaudeTextNode": ClaudeTextNode,
    "ClaudeVisionNode": ClaudeVisionNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ClaudeTextNode": "APNext Claude Text Generator",
    "ClaudeVisionNode": "APNext Claude Vision Analyzer",
}
