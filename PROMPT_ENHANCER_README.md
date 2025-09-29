# Gemini Prompt Enhancer Node

## Overview

The **GeminiPromptEnhancer** node is a powerful tool that transforms simple prompts into detailed, cinematic descriptions perfect for AI video generation models like Wan 2.2. It combines rule-based enhancement with AI-powered refinement using Google's Gemini models.

## Features

### 🎬 Cinematic Enhancement
- Adds professional cinematography terms based on the Wan 2.2 guide
- Includes lighting, camera angles, shot types, and visual effects
- Supports multiple enhancement modes for different creative needs

### 🤖 AI-Powered Refinement
- Uses Gemini models for sophisticated prompt enhancement
- Maintains the core subject while adding cinematic details
- Custom LLM prompts for specialized enhancement styles

### 🎯 Multiple Enhancement Modes

1. **Random Mix (4-6 elements)** - Randomly selects 4-6 cinematic elements
2. **Cinematic Focus** - Emphasizes shot composition and camera work
3. **Lighting Focus** - Concentrates on lighting and atmosphere
4. **Camera Focus** - Highlights camera angles and movements
5. **Motion Focus** - Emphasizes movement and character emotions
6. **Style Focus** - Focuses on visual style and effects
7. **Full Enhancement** - Uses all available categories
8. **LLM Only (No Random)** - Pure AI enhancement without random elements

## Input Parameters

### Required
- **base_prompt** (STRING): Your simple prompt (e.g., "A knight fighting a dragon")
- **enhancement_mode** (DROPDOWN): Select enhancement style
- **use_llm** (BOOLEAN): Enable/disable Gemini AI enhancement
- **gemini_model** (DROPDOWN): Choose Gemini model version
- **seed** (INT): Control randomization (0-999999)

### Optional
- **custom_llm_prompt** (STRING): Custom instructions for AI enhancement
- **intensity** (FLOAT): Control enhancement strength (0.1-2.0)

## Output

The node provides three outputs:
1. **enhanced_prompt** - Final enhanced prompt (combines random + AI if enabled)
2. **random_enhanced** - Random enhancement only
3. **llm_enhanced** - AI enhancement only

## Cinematic Categories

The node uses these professional cinematography categories:

### Visual Elements
- **Visual Style**: cinematic, hyper-realistic, 3D cartoon, pixel art, etc.
- **Lighting Type**: soft/hard lighting, edge lighting, silhouette, etc.
- **Light Source**: sunny, moonlighting, practical, firelight, etc.
- **Color Tone**: warm/cool colors, saturated/desaturated

### Camera Work
- **Camera Angle**: low/high angle, dutch angle, aerial shot
- **Shot Size**: close-up, medium shot, wide shot, establishing shot
- **Lens Type**: wide-angle, telephoto, fisheye
- **Camera Movement**: push in, pull back, pan, tilt, tracking shot

### Composition & Effects
- **Composition**: center, balanced, symmetrical, short-side
- **Motion**: running, dancing, flying, swimming, climbing
- **Character Emotion**: happy, mysterious, confident, vulnerable
- **Visual Effects**: motion blur, tilt-shift, time-lapse

## Usage Examples

### Basic Usage
```
Input: "A knight fighting a dragon"
Mode: Random Mix (4-6 elements)
Output: "A knight fighting a dragon, medium shot, edge lighting, warm colors, camera pans to the right, contemplative."
```

### Cinematic Focus
```
Input: "A woman walking through a forest"
Mode: Cinematic Focus
Output: "A woman walking through a forest, wide shot, low angle shot, soft lighting, center composition."
```

### LLM Enhancement
With AI enhancement enabled, the output becomes much more detailed:
```
"In a sweeping wide shot captured from a low angle, a determined woman moves through an ancient forest bathed in soft, dappled sunlight. The camera maintains a center composition as golden rays filter through towering trees, casting warm, ethereal light across the moss-covered forest floor. Her silhouette is gracefully framed against the luminous backdrop, with the lens capturing both the intimate human moment and the majestic scale of the natural cathedral surrounding her."
```

## Setup Requirements

1. **Environment Variable**: Set `GEMINI_API_KEY` with your Google AI API key
2. **Dependencies**: Requires `google-generativeai` package
3. **ComfyUI**: Install in ComfyUI custom nodes directory

## API Key Setup

### Windows
```batch
set GEMINI_API_KEY=your_api_key_here
```

### Linux/Mac
```bash
export GEMINI_API_KEY=your_api_key_here
```

## Integration with Workflows

The GeminiPromptEnhancer works seamlessly with other ComfyUI nodes:

1. **Text Input** → **GeminiPromptEnhancer** → **Video Generation Model**
2. **Image Analysis** → **GeminiPromptEnhancer** → **Style Transfer**
3. **Random Prompt Generator** → **GeminiPromptEnhancer** → **Batch Processing**

## Advanced Features

### Custom LLM Prompts
Create specialized enhancement styles by providing custom instructions:
```
"Enhance this prompt for a horror movie aesthetic with dark, moody lighting and unsettling camera work."
```

### Intensity Control
Adjust the enhancement strength:
- **0.1-0.5**: Subtle enhancement
- **1.0**: Standard enhancement
- **1.5-2.0**: Heavy enhancement with more elements

### Seed Control
Use consistent seeds for reproducible results across batch processing.

## Troubleshooting

### Common Issues
1. **API Key Error**: Ensure GEMINI_API_KEY is set correctly
2. **Model Not Found**: Check available Gemini models in dropdown
3. **Empty Output**: Verify internet connection for LLM enhancement

### Performance Tips
1. Use lower intensity for faster processing
2. Disable LLM for pure random enhancement
3. Use consistent seeds for batch workflows

## Examples and Workflows

See the `/examples` directory for:
- `prompt_enhancer_example.py` - Standalone testing script
- Sample ComfyUI workflows demonstrating integration
- Batch processing examples

## Contributing

To add new cinematic categories or enhancement modes:
1. Edit the `CINEMATIC_TERMS` dictionary in `sdxl_utility.py`
2. Add new modes to the `enhancement_modes` list
3. Implement the logic in `enhance_prompt_basic()`

---

*This node is part of the comfyui_dagthomas extension and follows the Wan 2.2 video generation guidelines for optimal compatibility.*
