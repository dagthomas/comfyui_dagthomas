# APNext Dynamic Nodes Generator
# 
# This file handles the generation of dynamic APNext nodes based on data categories.
# All static nodes have been migrated to the modular structure in nodes/.
# 
# This creates nodes like: ArchitecturePromptNode, ArtPromptNode, etc.
# Migration Status: 26/26 static nodes migrated (100% complete)
# 

import os
from .utils.constants import CUSTOM_CATEGORY, next_dir

# Import APNextNode from modular structure
try:
    from .nodes.prompt_generators.apnext_nodes import APNextNode
    
    # Generate dynamic nodes for each category
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
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
                "CATEGORY": CUSTOM_CATEGORY,
                "_subcategory": category,
            },
        )
        globals()[class_name] = new_class
        NODE_CLASS_MAPPINGS[class_name] = new_class
        NODE_DISPLAY_NAME_MAPPINGS[class_name] = (
            f"APNext {category.replace('_', ' ').title()}"
        )
        
    print(f"✅ Generated {len(NODE_CLASS_MAPPINGS)} dynamic APNext nodes")
    
except ImportError as e:
    print(f"⚠️ Warning: Could not import APNextNode for dynamic nodes: {e}")
    # Dynamic nodes will not be available
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}