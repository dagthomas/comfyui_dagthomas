# ComfyUI DagThomas Custom Nodes
# Modular node loading system

import os
import sys

# Import dynamic nodes from APNext file
try:
    from .apnext import NODE_CLASS_MAPPINGS as DYNAMIC_MAPPINGS
    from .apnext import NODE_DISPLAY_NAME_MAPPINGS as DYNAMIC_DISPLAY_MAPPINGS
    print(f"Loaded {len(DYNAMIC_MAPPINGS)} dynamic APNext nodes")
except Exception as e:
    print(f"Warning: Could not load dynamic nodes: {e}")
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
except Exception:
    pass

try:
    from .nodes.gemini.next_scene import GeminiNextScene
    NEW_MAPPINGS["GeminiNextScene"] = GeminiNextScene
    NEW_DISPLAY_MAPPINGS["GeminiNextScene"] = "APNext Gemini Next Scene"
except Exception:
    pass

try:
    from .nodes.gemini.text_only import GeminiTextOnly
    NEW_MAPPINGS["GeminiTextOnly"] = GeminiTextOnly
    NEW_DISPLAY_MAPPINGS["GeminiTextOnly"] = "APNext Gemini Text Only"
except Exception:
    pass

try:
    from .nodes.gemini.custom_vision import GeminiCustomVision
    NEW_MAPPINGS["GeminiCustomVision"] = GeminiCustomVision
    NEW_DISPLAY_MAPPINGS["GeminiCustomVision"] = "APNext Gemini Custom Vision"
except Exception:
    pass

# GPT Nodes
try:
    from .nodes.gpt.mini_node import GptMiniNode
    NEW_MAPPINGS["GptMiniNode"] = GptMiniNode
    NEW_DISPLAY_MAPPINGS["GptMiniNode"] = "APNext GPT Mini Generator"
except Exception:
    pass

# MiniCPM Nodes
try:
    from .nodes.minicpm.video_node import MiniCPMVideoNode
    NEW_MAPPINGS["MiniCPMVideoNode"] = MiniCPMVideoNode
    NEW_DISPLAY_MAPPINGS["MiniCPMVideoNode"] = "APNext MiniCPM Video"
except Exception:
    pass

try:
    from .nodes.minicpm.image_node import MiniCPMImageNode
    NEW_MAPPINGS["MiniCPMImageNode"] = MiniCPMImageNode
    NEW_DISPLAY_MAPPINGS["MiniCPMImageNode"] = "APNext MiniCPM Image"
except Exception:
    pass

try:
    from .nodes.string_utils.mixer import SentenceMixerNode
    NEW_MAPPINGS["SentenceMixerNode"] = SentenceMixerNode
    NEW_DISPLAY_MAPPINGS["SentenceMixerNode"] = "APNext Sentence Mixer"
except Exception:
    pass

try:
    from .nodes.string_utils.flexible_merger import FlexibleStringMergerNode
    NEW_MAPPINGS["FlexibleStringMergerNode"] = FlexibleStringMergerNode
    NEW_DISPLAY_MAPPINGS["FlexibleStringMergerNode"] = "APNext Flexible String Merger"
except Exception:
    pass

try:
    from .nodes.string_utils.merger import StringMergerNode
    NEW_MAPPINGS["StringMergerNode"] = StringMergerNode
    NEW_DISPLAY_MAPPINGS["StringMergerNode"] = "APNext String Merger"
except Exception:
    pass

try:
    from .nodes.string_utils.random_picker import RandomStringPicker
    NEW_MAPPINGS["RandomStringPicker"] = RandomStringPicker
    NEW_DISPLAY_MAPPINGS["RandomStringPicker"] = "APNext Random String Picker"
except Exception:
    pass

try:
    from .nodes.string_utils.string_input import StringInput
    NEW_MAPPINGS["StringInput"] = StringInput
    NEW_DISPLAY_MAPPINGS["StringInput"] = "APNext String Input"
except Exception:
    pass

try:
    from .nodes.file_utils.random_integer import RandomIntegerNode
    NEW_MAPPINGS["RandomIntegerNode"] = RandomIntegerNode
    NEW_DISPLAY_MAPPINGS["RandomIntegerNode"] = "APNext Random Integer Generator"
except Exception:
    pass

try:
    from .nodes.latent_generators.pgsd3_latent import PGSD3LatentGenerator
    NEW_MAPPINGS["PGSD3LatentGenerator"] = PGSD3LatentGenerator
    NEW_DISPLAY_MAPPINGS["PGSD3LatentGenerator"] = "APNext PGSD3LatentGenerator"
except Exception:
    pass

try:
    from .nodes.latent_generators.apn_latent import APNLatent
    NEW_MAPPINGS["APNLatent"] = APNLatent
    NEW_DISPLAY_MAPPINGS["APNLatent"] = "APNext Latent Generator"
except Exception:
    pass

try:
    from .nodes.prompt_generators.auto_prompter import PromptGenerator
    NEW_MAPPINGS["PromptGenerator"] = PromptGenerator
    NEW_DISPLAY_MAPPINGS["PromptGenerator"] = "Auto Prompter"
except Exception:
    pass

try:
    from .nodes.prompt_generators.apnext_nodes import APNextNode
    NEW_MAPPINGS["APNextNode"] = APNextNode
    NEW_DISPLAY_MAPPINGS["APNextNode"] = "APNext Node"
except Exception:
    pass

try:
    from .nodes.gpt.vision_cloner import GptVisionCloner
    NEW_MAPPINGS["GptVisionCloner"] = GptVisionCloner
    NEW_DISPLAY_MAPPINGS["GptVisionCloner"] = "APNext GPT Vision Cloner"
except Exception:
    pass

try:
    from .nodes.gpt.custom_vision import GptCustomVision
    NEW_MAPPINGS["GptCustomVision"] = GptCustomVision
    NEW_DISPLAY_MAPPINGS["GptCustomVision"] = "APNext GPT Custom Vision"
except Exception:
    pass

try:
    from .nodes.ollama.node import OllamaNode
    NEW_MAPPINGS["OllamaNode"] = OllamaNode
    NEW_DISPLAY_MAPPINGS["OllamaNode"] = "APNext OllamaNode"
except Exception:
    pass

try:
    from .nodes.ollama.vision_node import OllamaVisionNode
    NEW_MAPPINGS["OllamaVisionNode"] = OllamaVisionNode
    NEW_DISPLAY_MAPPINGS["OllamaVisionNode"] = "APNext OllamaVision"
except Exception:
    pass

try:
    from .nodes.phi.model_loader import PhiModelLoader
    NEW_MAPPINGS["PhiModelLoader"] = PhiModelLoader
    NEW_DISPLAY_MAPPINGS["PhiModelLoader"] = "APNext Phi Model Loader"
except Exception:
    pass

try:
    from .nodes.phi.inference import PhiModelInference
    NEW_MAPPINGS["PhiModelInference"] = PhiModelInference
    NEW_DISPLAY_MAPPINGS["PhiModelInference"] = "APNext Phi Model Inference"
except Exception:
    pass

try:
    from .nodes.phi.custom_inference import PhiCustomModelInference
    NEW_MAPPINGS["PhiCustomModelInference"] = PhiCustomModelInference
    NEW_DISPLAY_MAPPINGS["PhiCustomModelInference"] = "APNext Phi Custom Model Inference"
except Exception:
    pass

try:
    from .nodes.file_utils.file_reader import FileReaderNode
    NEW_MAPPINGS["FileReaderNode"] = FileReaderNode
    NEW_DISPLAY_MAPPINGS["FileReaderNode"] = "APNext Local random prompt"
except Exception:
    pass

try:
    from .nodes.file_utils.prompt_loader import CustomPromptLoader
    NEW_MAPPINGS["CustomPromptLoader"] = CustomPromptLoader
    NEW_DISPLAY_MAPPINGS["CustomPromptLoader"] = "APNext Custom Prompts"
except Exception:
    pass

# Universal Generator
try:
    from .nodes.prompt_generators.universal_generator import APNextGenerator
    NEW_MAPPINGS["APNextGenerator"] = APNextGenerator
    NEW_DISPLAY_MAPPINGS["APNextGenerator"] = "APNext Universal Generator"
except Exception:
    pass

# Universal Vision Cloner
try:
    from .nodes.prompt_generators.universal_vision_cloner import UniversalVisionCloner
    NEW_MAPPINGS["UniversalVisionCloner"] = UniversalVisionCloner
    NEW_DISPLAY_MAPPINGS["UniversalVisionCloner"] = "APNext Universal Vision Cloner"
except Exception:
    pass

# APNext FX Nodes
try:
    from .nodes.image_fx.bloom import APNextBloom
    NEW_MAPPINGS["APNextBloom"] = APNextBloom
    NEW_DISPLAY_MAPPINGS["APNextBloom"] = "APNext Bloom FX"
except Exception:
    pass

try:
    from .nodes.image_fx.sharpen import APNextSharpen
    NEW_MAPPINGS["APNextSharpen"] = APNextSharpen
    NEW_DISPLAY_MAPPINGS["APNextSharpen"] = "APNext Sharpen FX"
except Exception:
    pass

try:
    from .nodes.image_fx.noise import APNextNoise
    NEW_MAPPINGS["APNextNoise"] = APNextNoise
    NEW_DISPLAY_MAPPINGS["APNextNoise"] = "APNext Noise FX"
except Exception:
    pass

try:
    from .nodes.image_fx.rough import APNextRough
    NEW_MAPPINGS["APNextRough"] = APNextRough
    NEW_DISPLAY_MAPPINGS["APNextRough"] = "APNext Rough FX"
except Exception:
    pass

# Grok Nodes
try:
    from .nodes.grok.text_node import GrokTextNode
    NEW_MAPPINGS["GrokTextNode"] = GrokTextNode
    NEW_DISPLAY_MAPPINGS["GrokTextNode"] = "APNext Grok Text Generator"
except Exception:
    pass

try:
    from .nodes.grok.vision_node import GrokVisionNode
    NEW_MAPPINGS["GrokVisionNode"] = GrokVisionNode
    NEW_DISPLAY_MAPPINGS["GrokVisionNode"] = "APNext Grok Vision Analyzer"
except Exception:
    pass

# Claude Nodes
try:
    from .nodes.claude.text_node import ClaudeTextNode
    NEW_MAPPINGS["ClaudeTextNode"] = ClaudeTextNode
    NEW_DISPLAY_MAPPINGS["ClaudeTextNode"] = "APNext Claude Text Generator"
except Exception:
    pass

try:
    from .nodes.claude.vision_node import ClaudeVisionNode
    NEW_MAPPINGS["ClaudeVisionNode"] = ClaudeVisionNode
    NEW_DISPLAY_MAPPINGS["ClaudeVisionNode"] = "APNext Claude Vision Analyzer"
except Exception:
    pass

# Groq Nodes
try:
    from .nodes.groq.text_node import GroqTextNode
    NEW_MAPPINGS["GroqTextNode"] = GroqTextNode
    NEW_DISPLAY_MAPPINGS["GroqTextNode"] = "APNext Groq Text Generator"
except Exception:
    pass

try:
    from .nodes.groq.vision_node import GroqVisionNode
    NEW_MAPPINGS["GroqVisionNode"] = GroqVisionNode
    NEW_DISPLAY_MAPPINGS["GroqVisionNode"] = "APNext Groq Vision Analyzer"
except Exception:
    pass

# QwenVL Nodes
try:
    from .nodes.qwenvl.vision_node import QwenVLVisionNode
    NEW_MAPPINGS["QwenVLVisionNode"] = QwenVLVisionNode
    NEW_DISPLAY_MAPPINGS["QwenVLVisionNode"] = "APNext QwenVL Vision Analyzer"
except Exception:
    pass

try:
    from .nodes.qwenvl.vision_cloner import QwenVLVisionCloner
    NEW_MAPPINGS["QwenVLVisionCloner"] = QwenVLVisionCloner
    NEW_DISPLAY_MAPPINGS["QwenVLVisionCloner"] = "APNext QwenVL Vision Cloner"
except Exception:
    pass

try:
    from .nodes.qwenvl.video_node import QwenVLVideoNode
    NEW_MAPPINGS["QwenVLVideoNode"] = QwenVLVideoNode
    NEW_DISPLAY_MAPPINGS["QwenVLVideoNode"] = "APNext QwenVL Video Analyzer"
except Exception:
    pass

try:
    from .nodes.qwenvl.zimage_vision import QwenVLZImageVision
    NEW_MAPPINGS["QwenVLZImageVision"] = QwenVLZImageVision
    NEW_DISPLAY_MAPPINGS["QwenVLZImageVision"] = "APNext QwenVL Z-Image Vision Cloner"
except Exception:
    pass

try:
    from .nodes.qwenvl.next_scene import QwenVLNextScene
    NEW_MAPPINGS["QwenVLNextScene"] = QwenVLNextScene
    NEW_DISPLAY_MAPPINGS["QwenVLNextScene"] = "APNext QwenVL Next Scene"
except Exception:
    pass

try:
    from .nodes.qwenvl.frame_prep import QwenVLFramePrep
    NEW_MAPPINGS["QwenVLFramePrep"] = QwenVLFramePrep
    NEW_DISPLAY_MAPPINGS["QwenVLFramePrep"] = "APNext QwenVL Frame Prep"
except Exception:
    pass

# APNext Advanced FX Nodes
try:
    from .nodes.image_fx.color_grading import APNextColorGrading
    NEW_MAPPINGS["APNextColorGrading"] = APNextColorGrading
    NEW_DISPLAY_MAPPINGS["APNextColorGrading"] = "APNext Color Grading FX"
except Exception:
    pass

try:
    from .nodes.image_fx.cross_processing import APNextCrossProcessing
    NEW_MAPPINGS["APNextCrossProcessing"] = APNextCrossProcessing
    NEW_DISPLAY_MAPPINGS["APNextCrossProcessing"] = "APNext Cross Processing FX"
except Exception:
    pass

try:
    from .nodes.image_fx.split_toning import APNextSplitToning
    NEW_MAPPINGS["APNextSplitToning"] = APNextSplitToning
    NEW_DISPLAY_MAPPINGS["APNextSplitToning"] = "APNext Split Toning FX"
except Exception:
    pass

try:
    from .nodes.image_fx.hdr_tone_mapping import APNextHDRToneMapping
    NEW_MAPPINGS["APNextHDRToneMapping"] = APNextHDRToneMapping
    NEW_DISPLAY_MAPPINGS["APNextHDRToneMapping"] = "APNext HDR Tone Mapping FX"
except Exception:
    pass

try:
    from .nodes.image_fx.glitch_art import APNextGlitchArt
    NEW_MAPPINGS["APNextGlitchArt"] = APNextGlitchArt
    NEW_DISPLAY_MAPPINGS["APNextGlitchArt"] = "APNext Glitch Art FX"
except Exception:
    pass

try:
    from .nodes.image_fx.film_halation import APNextFilmHalation
    NEW_MAPPINGS["APNextFilmHalation"] = APNextFilmHalation
    NEW_DISPLAY_MAPPINGS["APNextFilmHalation"] = "APNext Film Halation FX"
except Exception:
    pass

# Combine mappings (modular nodes + dynamic nodes)
NODE_CLASS_MAPPINGS = {**NEW_MAPPINGS, **DYNAMIC_MAPPINGS}
NODE_DISPLAY_NAME_MAPPINGS = {**NEW_DISPLAY_MAPPINGS, **DYNAMIC_DISPLAY_MAPPINGS}

print(f"Loaded {len(NEW_MAPPINGS)} modular nodes, {len(DYNAMIC_MAPPINGS)} dynamic nodes")
print(f"Total: {len(NODE_CLASS_MAPPINGS)} nodes available")

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']