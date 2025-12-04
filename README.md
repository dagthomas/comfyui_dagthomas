# comfyui_dagthomas

### 您可以在这里找到中文信息
[plugin.aix.ink](https://plugin.aix.ink/archives/comfyui-dagthomas)

**Advanced Prompt Generation & Multi-Model AI Integration for ComfyUI**

A comprehensive suite of nodes for ComfyUI featuring multi-provider LLM support (OpenAI, Gemini, Claude, Grok, Groq, QwenVL), local model inference (Phi, MiniCPM, Ollama), professional image effects, and advanced prompt generation tools.

---

## 📦 Installation

### Method 1: ComfyUI Manager (Recommended)
Search for "comfyui_dagthomas" in ComfyUI Manager and click Install.

### Method 2: Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/dagthomas/comfyui_dagthomas
cd comfyui_dagthomas
pip install -r requirements.txt
```

---

## 🔑 API Key Configuration

Set your API keys as environment variables:

```bash
# OpenAI GPT
set OPENAI_API_KEY=sk-your-key-here

# Google Gemini
set GEMINI_API_KEY=your-key-here

# Anthropic Claude
set ANTHROPIC_API_KEY=your-key-here
# or
set CLAUDE_API_KEY=your-key-here

# xAI Grok
set XAI_API_KEY=your-key-here
# or
set GROK_API_KEY=your-key-here

# Groq
set GROQ_API_KEY=your-key-here
```

---

## 🧩 Node Categories

### 📝 Universal Nodes (Model-Agnostic)

#### APNext Universal Generator
**Display Name:** `APNext Universal Generator`

A model-agnostic prompt generator that automatically detects available API keys and supports all major LLM providers.

| Input | Description |
|-------|-------------|
| `input_text` | Base text to enhance |
| `model` | Select provider:model or "auto-detect" |
| `generation_mode` | Creative, Balanced, Focused, or Custom |
| `seed` | Seed for reproducible variations |
| `style_preference` | Cinematic, Photorealistic, Artistic, etc. |
| `detail_level` | Brief to Very Detailed output |

**Supported Models:**
- `gpt:gpt-4o`, `gpt:gpt-4o-mini`, `gpt:gpt-4-turbo`
- `gemini:gemini-2.5-flash`, `gemini:gemini-2.5-pro`
- `claude:claude-sonnet-4.5`, `claude:claude-3-5-sonnet`
- `grok:grok-beta`, `grok:grok-2-vision`
- `groq:llama-3.3-70b-versatile`

**Returns:** `(generated_prompt, model_used, seed_used)`

---

#### APNext Universal Vision Cloner
**Display Name:** `APNext Universal Vision Cloner`

Analyze images with any supported vision model to generate detailed descriptions or clone image styles.

| Input | Description |
|-------|-------------|
| `images` | One or more images to analyze |
| `model` | Vision model to use (auto-detect available) |
| `fade_percentage` | Blend percentage for multiple images |
| `analysis_mode` | Detailed Analysis, Style Cloning, Scene Description, Creative Interpretation |
| `output_format` | Text Only, JSON Structure, or Formatted Prompt |

**Returns:** `(formatted_output, raw_response, faded_image, model_used)`

---

### 🤖 Google Gemini Nodes

#### Gemini Prompt Enhancer
**Display Name:** `APNext Gemini Prompt Enhancer`

Enhances prompts with cinematic terminology and LLM refinement for video/image generation.

| Input | Description |
|-------|-------------|
| `base_prompt` | Original prompt to enhance |
| `enhancement_mode` | Random Mix, Cinematic/Lighting/Camera/Motion/Style Focus, Full Enhancement, or LLM Only |
| `use_llm` | Enable Gemini LLM enhancement |
| `intensity` | Enhancement intensity (0.1-2.0) |
| Optional dropdowns | visual_style, lighting_type, camera_angle, shot_size, lens_type, color_tone, etc. |

**Returns:** `(enhanced_prompt, random_enhanced, llm_enhanced)`

---

#### Gemini Custom Vision
**Display Name:** `APNext Gemini Custom Vision`

Analyze multiple images with custom prompts. Supports dynamic prompt templates with variable substitution.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `custom_prompt` | Custom analysis prompt |
| `dynamic_prompt` | Enable ##TAG##, ##SEX##, ##PRONOUNS##, ##WORDS## substitution |
| `fade_percentage` | Blend multiple images together |

**Returns:** `(output, clip_l, faded_image)`

---

#### Gemini Text Only
**Display Name:** `APNext Gemini Text Only`

Pure text generation with Gemini models. Supports dynamic prompt templates.

**Returns:** `(output, clip_l)`

---

#### Gemini Next Scene
**Display Name:** `APNext Gemini Next Scene`

Generate cinematic transitions for visual narratives. Creates the "next scene" based on a previous prompt and current frame.

| Input | Description |
|-------|-------------|
| `image` | Current frame image |
| `original_prompt` | Previous scene description |
| `focus_on` | Camera Movement, Framing Evolution, Environmental Reveals, Atmospheric Shifts |
| `transition_intensity` | Subtle, Moderate, or Dramatic |

**Returns:** `(next_scene_prompt, short_description)`

---

### 💬 OpenAI GPT Nodes

#### GPT Mini Generator
**Display Name:** `APNext GPT Mini Generator`

Efficient text generation using GPT-4o-mini.

| Input | Description |
|-------|-------------|
| `input_text` | Text to enhance |
| `happy_talk` | Enthusiastic vs professional tone |
| `compress` | Enable output compression |
| `poster` | Movie poster style formatting |

---

#### GPT Vision Cloner
**Display Name:** `APNext GPT Vision Cloner`

Clone image styles using GPT-4o vision capabilities with custom prompts.

---

#### GPT Custom Vision
**Display Name:** `APNext GPT Custom Vision`

Full custom vision analysis with GPT-4o.

---

### 🧠 Anthropic Claude Nodes

#### Claude Text Generator
**Display Name:** `APNext Claude Text Generator`

Text generation with Claude models (Claude 3.5 Sonnet, Claude Sonnet 4.5).

| Input | Description |
|-------|-------------|
| `input_text` | Text to process |
| `claude_model` | Model selection |
| `happy_talk`, `compress`, `poster` | Output style controls |
| `variation_instruction` | Custom instruction for creative variations |

---

#### Claude Vision Analyzer
**Display Name:** `APNext Claude Vision Analyzer`

Image analysis with Claude's multimodal capabilities.

---

### ⚡ xAI Grok Nodes

#### Grok Text Generator
**Display Name:** `APNext Grok Text Generator`

Text generation using xAI's Grok models.

---

#### Grok Vision Analyzer
**Display Name:** `APNext Grok Vision Analyzer`

Image analysis with Grok vision models.

---

### 🚀 Groq Nodes (Ultra-Fast Inference)

#### Groq Text Generator
**Display Name:** `APNext Groq Text Generator`

Lightning-fast text generation using Groq's optimized infrastructure with Llama and Mixtral models.

| Input | Description |
|-------|-------------|
| `groq_model` | llama-3.3-70b-versatile, llama-3.1-8b-instant, etc. |
| Other standard LLM inputs |

---

#### Groq Vision Analyzer
**Display Name:** `APNext Groq Vision Analyzer`

Fast image analysis with Groq vision models.

---

### 🔍 QwenVL Nodes (Local Vision)

#### QwenVL Vision Analyzer
**Display Name:** `APNext QwenVL Vision Analyzer`

Local vision analysis using Qwen-VL models. Downloads models automatically.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `qwen_model` | Qwen3-VL-4B-Instruct, etc. |
| `max_tokens` | Maximum response length |
| `keep_model_loaded` | Cache model in memory |

---

#### QwenVL Vision Cloner
**Display Name:** `APNext QwenVL Vision Cloner`

Clone image styles locally without API calls.

---

#### QwenVL Video Analyzer
**Display Name:** `APNext QwenVL Video Analyzer`

Analyze video content frame-by-frame.

---

### 🦙 Ollama Nodes (Local LLM)

#### Ollama Node
**Display Name:** `APNext OllamaNode`

Local LLM inference using Ollama. Supports any model installed in your Ollama instance.

| Input | Description |
|-------|-------------|
| `input_text` | Text to process |
| `model_name` | Any Ollama model (llama3, mistral, etc.) |
| `happy_talk`, `compress` | Output controls |

---

#### Ollama Vision
**Display Name:** `APNext OllamaVision`

Local vision analysis with Ollama multimodal models (llava, bakllava, etc.).

---

### 📸 MiniCPM Nodes (Local Vision)

#### MiniCPM Image Node
**Display Name:** `APNext MiniCPM Image`

Image understanding with MiniCPM-V 4.5 (OpenBMB). Supports thinking mode for complex reasoning.

| Input | Description |
|-------|-------------|
| `images` | Input images |
| `question` | Question about the image |
| `enable_thinking` | Deep reasoning mode |
| `precision` | bfloat16 or float16 |
| `unload_after_inference` | Free memory after use |

---

#### MiniCPM Video Node
**Display Name:** `APNext MiniCPM Video`

Video understanding and analysis.

---

### 🔬 Phi Nodes (Microsoft Vision)

#### Phi Model Loader
**Display Name:** `APNext Phi Model Loader`

Load Microsoft Phi-3.5-vision-instruct model.

| Input | Description |
|-------|-------------|
| `model_version` | Phi-3.5-vision-instruct |
| `image_crops` | 4 or 16 crops for detail |
| `attention_mechanism` | flash_attention_2, sdpa, or eager |

---

#### Phi Model Inference / Custom Inference
**Display Name:** `APNext Phi Model Inference`

Run inference with loaded Phi model.

---

### 🎨 Image FX Nodes

Professional image effects using optimized tensor operations.

#### APNext Bloom FX
Creates a bloom/glow effect on bright areas.

| Input | Description |
|-------|-------------|
| `intensity` | Bloom strength (0-5) |
| `threshold` | Brightness threshold (0-1) |
| `blur_radius` | Glow spread (1-50) |
| `blend_mode` | additive, screen, or overlay |

---

#### APNext Color Grading FX
Professional color grading with LUT support or manual controls.

| Input | Description |
|-------|-------------|
| `method` | manual or lut_file |
| `lut_file` | .cube, .3dl, or image LUT |
| `exposure` | -3 to +3 stops |
| `contrast`, `saturation` | Standard adjustments |
| `highlights`, `shadows` | Tone controls |
| `temperature`, `tint` | White balance |

**Supported LUT Formats:** .cube (Adobe/Blackmagic), .3dl (Autodesk/Flame), Image LUTs (.png, .jpg)

---

#### APNext Sharpen FX
Intelligent image sharpening.

---

#### APNext Noise FX
Add film grain and noise effects.

---

#### APNext Rough FX
Add texture and roughness.

---

#### APNext Cross Processing FX
Film cross-processing color effects.

---

#### APNext Split Toning FX
Separate color toning for highlights and shadows.

---

#### APNext HDR Tone Mapping FX
HDR-style tone mapping.

---

#### APNext Glitch Art FX
Digital glitch and databending effects.

---

#### APNext Film Halation FX
Classic film halation (light bleeding) effect.

---

### 📐 Latent Generators

#### APNext Latent Generator
**Display Name:** `APNext Latent Generator`

Generate latent tensors with intelligent dimension calculation.

| Input | Description |
|-------|-------------|
| `width`, `height` | Base dimensions (0 = auto-calculate) |
| `megapixel_scale` | Target megapixels (0.1-2.0) |
| `aspect_ratio` | 1:1, 3:2, 4:3, 16:9, 21:9 |
| `is_portrait` | Portrait orientation |

**Returns:** `(LATENT, width, height)`

---

#### PGSD3 Latent Generator
**Display Name:** `APNext PGSD3LatentGenerator`

Optimized latent generation for Stable Diffusion 3 pipelines.

---

### 🎲 Prompt Generators

#### Auto Prompter
**Display Name:** `Auto Prompter`

Generate random prompts from extensive category databases.

| Input | Description |
|-------|-------------|
| `subject` | Main subject (can include LoRA triggers) |
| `custom` | Prefix text for styling |
| `artform` | Photography, digital art, etc. |
| Various category selections | Random or specific choices |

---

#### APNext Node
**Display Name:** `APNext Node`

Advanced prompt building with category-based enhancements.

### Overview

![Node Family Overview](https://github.com/user-attachments/assets/89c23e6f-44f5-4d2f-bb37-abf8cbd797c4)

The system includes numerous nodes that can be chained together to create complex workflows:

![Node Chaining Example](https://github.com/user-attachments/assets/bf402844-ffdc-4dcf-bc6c-28d40e125011)

Supports **24 main categories** with subcategories:
- **Architecture:** styles, buildings, interiors, materials
- **Art:** painting, sculpture, techniques, palettes
- **Artist:** concept artists, illustrators, painters
- **Character:** anime, fantasy, sci-fi, superheroes
- **Cinematic:** directors, genres, effects, color grading
- **Fashion:** designers, outfits, accessories
- **Feelings:** emotional modifiers
- **Geography:** countries, nationalities
- **Human:** jobs, hobbies, groups
- **Interaction:** individual, couple, group, crowd interactions
- **Keywords:** modifiers, genres, trending terms
- **People:** archetypes, body types, expressions
- **Photography:** cameras, lenses, lighting, film types
- **Plots:** action, romance, horror, sci-fi scenarios
- **Poses:** portrait and action poses
- **Scene:** weather, textures, environments
- **Science:** astronomy, mathematics, medical
- **Stuff:** seasonal objects, gadgets, fantasy items
- **Time:** eras, decades, centuries
- **Typography:** fonts, word art styles
- **Vehicle:** cars, classic cars, vehicle types
- **Video Game:** games, engines, actions

---

### 🔧 Utility Nodes

#### String Merger
**Display Name:** `APNext String Merger`

Combine multiple strings with separators.

---

#### Flexible String Merger
**Display Name:** `APNext Flexible String Merger`

Advanced string combining with custom formatting.

---

#### Sentence Mixer
**Display Name:** `APNext Sentence Mixer`

Shuffle and mix sentences from multiple inputs for creative variations.

---

#### Custom Prompt Loader
**Display Name:** `APNext Custom Prompts`

Load prompt templates from the `data/custom_prompts/` directory.

Included templates:
- `promptcreator.txt` - Full creative prompt generation
- `image_analyze.txt` - Image analysis prompts
- `gemini_video.txt` - Video generation prompts
- `cloner.txt` - Style cloning prompts
- Various LoRA-specific templates (ohwx, t5xxl, etc.)

---

#### Local Random Prompt
**Display Name:** `APNext Local random prompt`

Load random prompts from local text files.

---

#### Random Integer Generator
**Display Name:** `APNext Random Integer Generator`

Generate random integers with min/max range.

---

## 📁 Adding Custom Categories

Create your own categories for APNextNode:

1. Create a folder in `data/next/` (e.g., `data/next/mycategory/`)
2. Add JSON files for each field

### Simple Format
```json
["item1", "item2", "item3"]
```

### Advanced Format
```json
{
  "preprompt": "with",
  "separator": " and ",
  "endprompt": "visual effects",
  "items": ["motion blur", "lens flare", "particle effects"],
  "attributes": {
    "motion blur": ["dynamic", "cinematic"],
    "lens flare": ["bright", "atmospheric"]
  }
}
```

---

## 📝 Custom Prompt Templates

Create your own prompt templates for use with the **Custom Prompt Loader** node.

### Location
Place `.txt` files in: `data/custom_prompts/`

### Creating a Template

Templates are plain text files containing instructions for LLM nodes. They support dynamic variable substitution:

| Variable | Description |
|----------|-------------|
| `##TAG##` | Replaced with the `tag` input (e.g., "ohwx man") |
| `##SEX##` | Replaced with the `sex` input (e.g., "male", "female") |
| `##PRONOUNS##` | Replaced with pronouns (e.g., "him, his") |
| `##WORDS##` | Replaced with target word count |

### Example Template

Create a file `data/custom_prompts/my_style.txt`:

```
As a professional art critic, describe the provided image in detail.
Focus on creating a cohesive scene as if describing a movie still.

If the subject is ##TAG##, use ##PRONOUNS## pronouns appropriately.
The subject is ##SEX##.

Include:
- Main subject description with clothing, accessories, position
- Setting and environment details
- Lighting type, direction, and atmosphere
- Color palette and emotional tone
- Camera angle and composition

Output approximately ##WORDS## words.
Do not use JSON format. Provide a single cohesive paragraph.
```

### Included Templates

| Template | Purpose |
|----------|---------|
| `promptcreator.txt` | Detailed image analysis (~150 words) |
| `promptcreator_small.txt` | Concise image analysis |
| `image_analyze.txt` | General image description |
| `cloner.txt` | Style cloning prompts |
| `gemini_video.txt` | Video generation prompts |
| `gemini_ohwx.txt` | LoRA trigger-aware prompts |
| `t5xxl.txt` | T5-XXL optimized prompts |
| `ltxv.txt` | LTX Video model prompts |
| `next_scene.txt` | Cinematic scene transitions |

---

## ⚙️ Configuring LLM Models

Customize available models by editing JSON configuration files in the `data/` folder.

### Model Configuration Files

| File | Provider | Description |
|------|----------|-------------|
| `gemini_models.json` | Google Gemini | Gemini model list |
| `gpt_models.json` | OpenAI | GPT model list |
| `claude_models.json` | Anthropic | Claude model list |
| `grok_models.json` | xAI | Grok model list |
| `groq_models.json` | Groq | Groq model list (text + vision) |
| `qwenvl_models.json` | QwenVL | Local Qwen vision models |

### Basic Format

Most model files use a simple array format:

```json
{
    "models": [
        "model-name-1",
        "model-name-2",
        "model-name-3"
    ]
}
```

### Example: Adding New Gemini Models

Edit `data/gemini_models.json`:

```json
{
    "models": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-flash-latest",
        "gemini-flash-lite-latest",
        "gemini-2.5-flash-lite",
        "gemini-exp-1206"
    ]
}
```

### Example: Adding New Claude Models

Edit `data/claude_models.json`:

```json
{
    "models": [
        "claude-sonnet-4.5",
        "claude-sonnet-4",
        "claude-sonnet-3.7",
        "claude-opus-4.1",
        "claude-opus-4",
        "claude-haiku-3.5",
        "claude-haiku-3"
    ]
}
```

### Groq Models (Advanced Format)

Groq supports separate text and vision model lists:

```json
{
    "text_models": [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "groq/compound",
        "qwen/qwen3-32b"
    ],
    "vision_models": [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct"
    ],
    "note": "Edit this file to add/remove models"
}
```

### Notes

- **Restart ComfyUI** after editing model configuration files
- For Groq, the system will first try to fetch models from the API, then fall back to the JSON file
- Model names must match exactly what the provider's API expects
- Invalid model names will cause API errors at runtime

---

## 🖼️ Example Workflows

Example workflows are available in the `examples/` directory:

- **APNext workflows:** `examples/flux/apnext/`
- **Florence2 local:** `examples/flux/florence2/`
- **GPT-4o Vision:** `examples/flux/gpt-4o_vision/`
- **Ollama local:** `examples/flux/ollama_local_llm/`
- **MiniCPM:** `examples/minicpm/`

---

## 📋 Requirements

```
Pillow>=10.4.0
requests>=2.32.5
openai>=1.44.0
blend-modes>=2.1.0
huggingface_hub>=0.34.0
color_matcher>=0.5.0
chardet>=5.2.0
google-generativeai>=0.7.2
anthropic
transformers>=4.40.0
decord>=0.6.0
scipy>=1.10.0
tqdm>=4.67.1
```

---

## 🔄 Model Support Matrix

| Provider | Text | Vision | Video | Local |
|----------|------|--------|-------|-------|
| OpenAI GPT | ✅ | ✅ | ❌ | ❌ |
| Google Gemini | ✅ | ✅ | ✅ | ❌ |
| Anthropic Claude | ✅ | ✅ | ❌ | ❌ |
| xAI Grok | ✅ | ✅ | ❌ | ❌ |
| Groq | ✅ | ✅ | ❌ | ❌ |
| QwenVL | ✅ | ✅ | ✅ | ✅ |
| Ollama | ✅ | ✅ | ❌ | ✅ |
| MiniCPM | ✅ | ✅ | ✅ | ✅ |
| Phi-3.5 | ✅ | ✅ | ❌ | ✅ |

---

## 📝 License

MIT License

---

## 🙏 Acknowledgments

Built for the ComfyUI community. Special thanks to all contributors and users providing feedback.
