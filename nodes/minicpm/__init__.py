# MiniCPM-V Node
from .video_node import MiniCPMVideoNode
from .image_node import MiniCPMImageNode

NODE_CLASS_MAPPINGS = {
    "MiniCPMVideoNode": MiniCPMVideoNode,
    "MiniCPMImageNode": MiniCPMImageNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniCPMVideoNode": "APNext MiniCPM Video",
    "MiniCPMImageNode": "APNext MiniCPM Image",
}

