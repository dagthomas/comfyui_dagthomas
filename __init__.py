# ComfyUI DagThomas Custom Nodes
# Modular node loading system

import os
import sys

# Add the package to Python path
package_dir = os.path.dirname(__file__)
if package_dir not in sys.path:
    sys.path.append(package_dir)

# Import dynamic nodes from APNext file
try:
    from .apnext import NODE_CLASS_MAPPINGS as DYNAMIC_MAPPINGS
    from .apnext import NODE_DISPLAY_NAME_MAPPINGS as DYNAMIC_DISPLAY_MAPPINGS
    print(f"✅ Loaded {len(DYNAMIC_MAPPINGS)} dynamic APNext nodes")
except Exception as e:
    print(f"⚠️ Could not load dynamic nodes: {e}")
    DYNAMIC_MAPPINGS = {}
    DYNAMIC_DISPLAY_MAPPINGS = {}

# Import from new modular structure
NEW_MAPPINGS = {}
NEW_DISPLAY_MAPPINGS = {}

# Try to import new modular nodes (graceful fallback if not available)

# Gemini Nodes
try:
    from .nodes.gemini.prompt_enhancer import GeminiPromptEnhancer
    NEW_MAPPINGS["GeminiPromptEnhancer"] = GeminiPromptEnhancer
    NEW_DISPLAY_MAPPINGS["GeminiPromptEnhancer"] = "APNext Gemini Prompt Enhancer"
    print("✅ Loaded modular GeminiPromptEnhancer")
except Exception as e:
    print(f"⚠️ Could not load modular GeminiPromptEnhancer: {e}")

try:
    from .nodes.gemini.next_scene import GeminiNextScene
    NEW_MAPPINGS["GeminiNextScene"] = GeminiNextScene
    NEW_DISPLAY_MAPPINGS["GeminiNextScene"] = "APNext Gemini Next Scene"
    print("✅ Loaded modular GeminiNextScene")
except Exception as e:
    print(f"⚠️ Could not load modular GeminiNextScene: {e}")

try:
    from .nodes.gemini.text_only import GeminiTextOnly
    NEW_MAPPINGS["GeminiTextOnly"] = GeminiTextOnly
    NEW_DISPLAY_MAPPINGS["GeminiTextOnly"] = "APNext Gemini Text Only"
    print("✅ Loaded modular GeminiTextOnly")
except Exception as e:
    print(f"⚠️ Could not load modular GeminiTextOnly: {e}")

try:
    from .nodes.gemini.custom_vision import GeminiCustomVision
    NEW_MAPPINGS["GeminiCustomVision"] = GeminiCustomVision
    NEW_DISPLAY_MAPPINGS["GeminiCustomVision"] = "APNext Gemini Custom Vision"
    print("✅ Loaded modular GeminiCustomVision")
except Exception as e:
    print(f"⚠️ Could not load modular GeminiCustomVision: {e}")

# GPT Nodes
try:
    from .nodes.gpt.mini_node import GptMiniNode
    NEW_MAPPINGS["GptMiniNode"] = GptMiniNode
    NEW_DISPLAY_MAPPINGS["GptMiniNode"] = "APNext GPT Mini Generator"
    print("✅ Loaded modular GptMiniNode")
except Exception as e:
    print(f"⚠️ Could not load modular GptMiniNode: {e}")


try:
    from .nodes.string_utils.mixer import SentenceMixerNode
    NEW_MAPPINGS["SentenceMixerNode"] = SentenceMixerNode
    NEW_DISPLAY_MAPPINGS["SentenceMixerNode"] = "APNext Sentence Mixer"
    print("✅ Loaded modular SentenceMixerNode")
except Exception as e:
    print(f"⚠️ Could not load modular SentenceMixerNode: {e}")

try:
    from .nodes.string_utils.flexible_merger import FlexibleStringMergerNode
    NEW_MAPPINGS["FlexibleStringMergerNode"] = FlexibleStringMergerNode
    NEW_DISPLAY_MAPPINGS["FlexibleStringMergerNode"] = "APNext Flexible String Merger"
    print("✅ Loaded modular FlexibleStringMergerNode")
except Exception as e:
    print(f"⚠️ Could not load modular FlexibleStringMergerNode: {e}")

try:
    from .nodes.string_utils.merger import StringMergerNode
    NEW_MAPPINGS["StringMergerNode"] = StringMergerNode
    NEW_DISPLAY_MAPPINGS["StringMergerNode"] = "APNext String Merger"
    print("✅ Loaded modular StringMergerNode")
except Exception as e:
    print(f"⚠️ Could not load modular StringMergerNode: {e}")

try:
    from .nodes.file_utils.random_integer import RandomIntegerNode
    NEW_MAPPINGS["RandomIntegerNode"] = RandomIntegerNode
    NEW_DISPLAY_MAPPINGS["RandomIntegerNode"] = "APNext Random Integer Generator"
    print("✅ Loaded modular RandomIntegerNode")
except Exception as e:
    print(f"⚠️ Could not load modular RandomIntegerNode: {e}")

try:
    from .nodes.latent_generators.pgsd3_latent import PGSD3LatentGenerator
    NEW_MAPPINGS["PGSD3LatentGenerator"] = PGSD3LatentGenerator
    NEW_DISPLAY_MAPPINGS["PGSD3LatentGenerator"] = "APNext PGSD3LatentGenerator"
    print("✅ Loaded modular PGSD3LatentGenerator")
except Exception as e:
    print(f"⚠️ Could not load modular PGSD3LatentGenerator: {e}")

try:
    from .nodes.latent_generators.apn_latent import APNLatent
    NEW_MAPPINGS["APNLatent"] = APNLatent
    NEW_DISPLAY_MAPPINGS["APNLatent"] = "APNext Latent Generator"
    print("✅ Loaded modular APNLatent")
except Exception as e:
    print(f"⚠️ Could not load modular APNLatent: {e}")

try:
    from .nodes.prompt_generators.auto_prompter import PromptGenerator
    NEW_MAPPINGS["PromptGenerator"] = PromptGenerator
    NEW_DISPLAY_MAPPINGS["PromptGenerator"] = "Auto Prompter"
    print("✅ Loaded modular PromptGenerator")
except Exception as e:
    print(f"⚠️ Could not load modular PromptGenerator: {e}")

try:
    from .nodes.prompt_generators.apnext_nodes import APNextNode
    NEW_MAPPINGS["APNextNode"] = APNextNode
    NEW_DISPLAY_MAPPINGS["APNextNode"] = "APNext Node"
    print("✅ Loaded modular APNextNode")
except Exception as e:
    print(f"⚠️ Could not load modular APNextNode: {e}")

try:
    from .nodes.gpt.vision_cloner import GptVisionCloner
    NEW_MAPPINGS["GptVisionCloner"] = GptVisionCloner
    NEW_DISPLAY_MAPPINGS["GptVisionCloner"] = "APNext GPT Vision Cloner"
    print("✅ Loaded modular GptVisionCloner")
except Exception as e:
    print(f"⚠️ Could not load modular GptVisionCloner: {e}")

try:
    from .nodes.gpt.custom_vision import GptCustomVision
    NEW_MAPPINGS["GptCustomVision"] = GptCustomVision
    NEW_DISPLAY_MAPPINGS["GptCustomVision"] = "APNext GPT Custom Vision"
    print("✅ Loaded modular GptCustomVision")
except Exception as e:
    print(f"⚠️ Could not load modular GptCustomVision: {e}")

try:
    from .nodes.ollama.node import OllamaNode
    NEW_MAPPINGS["OllamaNode"] = OllamaNode
    NEW_DISPLAY_MAPPINGS["OllamaNode"] = "APNext OllamaNode"
    print("✅ Loaded modular OllamaNode")
except Exception as e:
    print(f"⚠️ Could not load modular OllamaNode: {e}")

try:
    from .nodes.ollama.vision_node import OllamaVisionNode
    NEW_MAPPINGS["OllamaVisionNode"] = OllamaVisionNode
    NEW_DISPLAY_MAPPINGS["OllamaVisionNode"] = "APNext OllamaVision"
    print("✅ Loaded modular OllamaVisionNode")
except Exception as e:
    print(f"⚠️ Could not load modular OllamaVisionNode: {e}")

try:
    from .nodes.phi.model_loader import PhiModelLoader
    NEW_MAPPINGS["PhiModelLoader"] = PhiModelLoader
    NEW_DISPLAY_MAPPINGS["PhiModelLoader"] = "APNext Phi Model Loader"
    print("✅ Loaded modular PhiModelLoader")
except Exception as e:
    print(f"⚠️ Could not load modular PhiModelLoader: {e}")

try:
    from .nodes.phi.inference import PhiModelInference
    NEW_MAPPINGS["PhiModelInference"] = PhiModelInference
    NEW_DISPLAY_MAPPINGS["PhiModelInference"] = "APNext Phi Model Inference"
    print("✅ Loaded modular PhiModelInference")
except Exception as e:
    print(f"⚠️ Could not load modular PhiModelInference: {e}")

try:
    from .nodes.phi.custom_inference import PhiCustomModelInference
    NEW_MAPPINGS["PhiCustomModelInference"] = PhiCustomModelInference
    NEW_DISPLAY_MAPPINGS["PhiCustomModelInference"] = "APNext Phi Custom Model Inference"
    print("✅ Loaded modular PhiCustomModelInference")
except Exception as e:
    print(f"⚠️ Could not load modular PhiCustomModelInference: {e}")

try:
    from .nodes.file_utils.file_reader import FileReaderNode
    NEW_MAPPINGS["FileReaderNode"] = FileReaderNode
    NEW_DISPLAY_MAPPINGS["FileReaderNode"] = "APNext Local random prompt"
    print("✅ Loaded modular FileReaderNode")
except Exception as e:
    print(f"⚠️ Could not load modular FileReaderNode: {e}")

try:
    from .nodes.file_utils.prompt_loader import CustomPromptLoader
    NEW_MAPPINGS["CustomPromptLoader"] = CustomPromptLoader
    NEW_DISPLAY_MAPPINGS["CustomPromptLoader"] = "APNext Custom Prompts"
    print("✅ Loaded modular CustomPromptLoader")
except Exception as e:
    print(f"⚠️ Could not load modular CustomPromptLoader: {e}")

# Universal Generator
try:
    from .nodes.prompt_generators.universal_generator import APNextGenerator
    NEW_MAPPINGS["APNextGenerator"] = APNextGenerator
    NEW_DISPLAY_MAPPINGS["APNextGenerator"] = "APNext Universal Generator"
    print("✅ Loaded modular APNextGenerator")
except Exception as e:
    print(f"⚠️ Could not load modular APNextGenerator: {e}")

# Universal Vision Cloner
try:
    from .nodes.prompt_generators.universal_vision_cloner import UniversalVisionCloner
    NEW_MAPPINGS["UniversalVisionCloner"] = UniversalVisionCloner
    NEW_DISPLAY_MAPPINGS["UniversalVisionCloner"] = "APNext Universal Vision Cloner"
    print("✅ Loaded modular UniversalVisionCloner")
except Exception as e:
    print(f"⚠️ Could not load modular UniversalVisionCloner: {e}")

# Combine mappings (modular nodes + dynamic nodes)
NODE_CLASS_MAPPINGS = {**NEW_MAPPINGS, **DYNAMIC_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NEW_DISPLAY_MAPPINGS, **DYNAMIC_DISPLAY_MAPPINGS}

print(f"📦 Loaded {len(NEW_MAPPINGS)} modular nodes, {len(DYNAMIC_MAPPINGS)} dynamic nodes")
print(f"🎉 Total: {len(NODE_CLASS_MAPPINGS)} nodes available")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']