# comfyui_dagthomas
### 您可以在这里找到中文信息
[plugin.aix.ink](https://plugin.aix.ink/archives/comfyui-dagthomas)

### Advanced Prompt Generation and Image Analysis

This plugin extends ComfyUI with advanced prompt generation capabilities and image analysis using GPT-4 Vision. It includes the following components:

## Classes

### 1. PromptGenerator

A versatile prompt generator for text-to-image AI systems.

Features:
- Generates prompts based on various customizable parameters
- Supports different art forms, photography styles, and digital art
- Allows for random selection or specific choices for each parameter
- Outputs separate prompts for different model components (e.g., CLIP, T5)

#### "subject" input field can be changed to support your style or lora, just add your subject and it will overwrite "man, woman ..." etc.

#### "custom" input field will add a prompt to the start of the prompt string. For loading styles

![image](https://github.com/dagthomas/comfyui_dagthomas/assets/4311672/2c6e7418-51a6-465c-8573-36f36300e8a6)

### GPT4VisionNode

Analyzes images using OpenAI's GPT-4 Vision model.

Features:
- Accepts image input and generates detailed descriptions
- Supports custom base prompts
- Offers options for "happy talk" (detailed descriptions) or simple outputs
- Includes compression options to limit output length
- Ability to create posters
  
![gpt-4o vision](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/gpt-4o_vision/dagthomas_gpt-4o-vision-workflow.png)
[Workflow](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/gpt-4o_vision/dagthomas_gpt4o_vision_workflow.json)

There is a toggle to create movie posters (08/04/24)
![ComfyUI_00161_](https://github.com/user-attachments/assets/262c7d5b-9770-492f-89b1-5dc6b62c9cc3)



### GPT4MiniNode

Generates text using OpenAI's GPT-4 model based on input text.

Features:
- Accepts text input and generates enhanced descriptions
- Supports custom base prompts
- Offers options for "happy talk" (detailed descriptions) or simple outputs
- Includes compression options to limit output length

![gpt-4o-mini](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/gpt-4o-mini/dagthomas_gpt-4o-mini_workflow.png)
[Workflow](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/gpt-4o-mini/dagthomas_gpt-4o-mini_workflow.json)



### OllamaNode

Generates text using custom Ollama based on input text.

Features:
- Accepts text input and generates enhanced descriptions
- Supports custom base prompts
- Offers options for "happy talk" (detailed descriptions) or simple outputs
- Includes compression options to limit output length

![Ollama](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/ollama_local_llm/comfyui_dagthomas_localllm__00044_.png)
[Workflow](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/ollama_local_llm/dagthomas_ollama_workflow.json)


### Pure Florence workflow: 

- You can also use a pure local Florence workflow without any of the others. The prompt will have some bloat, but works fine with Flux

![Florence2](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/florence2/dagthomas_florence2_workflow.png)
[Workflow](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/florence2/dagthomas_florence2_workflow.json)




### PGSD3LatentGenerator

Generates latent representations for use in Stable Diffusion 3 pipelines.

Features:
- Creates latent tensors with specified dimensions
- Supports batch processing
- Automatically adjusts dimensions to maintain a consistent megapixel count

![image](https://github.com/user-attachments/assets/b4e0bb6e-fded-4a99-b8d4-558f21124863)


## Usage

These classes can be integrated into ComfyUI workflows to enhance prompt generation, image analysis, and latent space manipulation for advanced AI image generation pipelines.

## Requirements

- OpenAI API key (for GPT4VisionNode and GPT4MiniNode)
- ComfyUI environment
- Additional dependencies as specified in the import statements

![354727214-63e0ecbc-650a-4bf1-bfca-d96a4a2a5f33](https://github.com/user-attachments/assets/45dda70b-8f1b-4615-a0c4-0b1c16ff94bc)


## Notes

- Ensure that your OpenAI API key is set in the environment variables
- Some classes may require additional data files (JSON) for their functionality
- Refer to the individual class documentation for specific usage instructions and input types


## Following were generated with Flux Dev - 03/08/2024
![ComfyUI_00005_](https://github.com/user-attachments/assets/d760bb22-797e-441e-a5b5-52e793a2b7c8)
![ComfyUI_00007_](https://github.com/user-attachments/assets/6975521b-85e3-4e18-a061-8cefb95159e5)
![ComfyUI_00034_](https://github.com/user-attachments/assets/470426d5-9320-40a3-816e-9d4bcfda6940)

## Following were generated with SD3 Medium - 06/15/2024

![image](https://github.com/dagthomas/comfyui_dagthomas/assets/4311672/94c76273-0a16-450a-876c-9eb515d995d5)
![image](https://github.com/dagthomas/comfyui_dagthomas/assets/4311672/37924320-6b46-48fb-9c5d-a24da2d3fd4c)

# ADVANCED PROMPTS

## APNext workflow example
[Workflow](https://github.com/dagthomas/comfyui_dagthomas/blob/master/examples/flux/apnext/APNext_Examples.json)

## APNextNode Function

The `APNextNode` is a custom node class designed for processing and enhancing prompts with additional contextual information. It's particularly useful for generating creative content by incorporating random elements from predefined categories.

## Features

- Processes input prompts and adds context from various categories
- Supports multiple input types including required and optional parameters
- Dynamically loads category data from JSON files
- Provides options for random selection of items within categories
- Allows for attribute addition to selected items

## Categories

The function supports multiple categories, which are loaded from JSON files in a specific directory structure. Based on the provided image, the supported categories include:

1. architecture
2. art
3. artist
4. brands
5. character
6. cinematic
7. fashion
8. feelings
9. geography
10. human
11. interaction
12. keywords
13. people
14. photography
15. plots
16. poses
17. scene
18. science
19. stuff
20. time
21. typography
22. vehicle
23. video_game

Each category can contain its own set of items and attributes, which are used to enhance the input prompt.

## Usage

The `APNextNode` class is designed to be used within a larger system, likely a node-based content generation pipeline. It processes input prompts and optional category selections to produce an enhanced prompt and a random output.

### Input Types

- **Required**:
  - `prompt`: A multiline string input for the base prompt
  - `separator`: A string to separate added elements (default: ",")

- **Optional**:
  - `string`: An additional string input (default: "")
  - `seed`: An integer seed for random operations (default: 0)
  - `attributes`: A boolean to toggle attribute inclusion (default: False)
  - Category-specific inputs: Dynamically generated based on available JSON files

### Output

The function returns two strings:
1. An enhanced prompt incorporating selected category items
2. A random output string with additional category items

## File Structure

The function expects a specific file structure for category data:

```
data/
└── next/
    └── [CATEGORY_NAME]/
        └── [field_name].json
```

Each JSON file should contain either an array of items or a dictionary with "items", "preprompt", "separator", "endprompt", and "attributes" keys.

## Note

This README provides an overview of the `APNextNode` function based on the given code snippet. For full implementation details and integration instructions, please refer to the complete source code and any additional documentation provided with the system where this node is used.

# Adding Custom Categories to APNextNode

The `APNextNode` function is designed to be flexible and allow users to add their own categories and fields. This guide will explain how to do this and how to structure the JSON files for new categories.

## Adding a New Category

1. Create a new folder in the `data/next/` directory. The folder name should be lowercase and represent your new category (e.g., `data/next/mycategory/`).

2. Inside this new folder, create one or more JSON files. Each JSON file represents a field within your category. The file name (without the .json extension) will be used as the field name in the `APNextNode` function.

## JSON Structure

The JSON file for each field can have two different structures:

### Simple Structure
A simple array of items:

```json
[
  "item1",
  "item2",
  "item3"
]
```

### Advanced Structure
A more detailed structure with additional properties:

```json
{
  "preprompt": "Optional text to appear before the selected items",
  "separator": ", ",
  "endprompt": "Optional text to appear after the selected items",
  "items": [
    "item1",
    "item2",
    "item3"
  ],
  "attributes": {
    "item1": ["attribute1", "attribute2"],
    "item2": ["attribute3", "attribute4"]
  }
}
```

#### Field Descriptions:
- `preprompt`: (Optional) Text that appears before the selected items.
- `separator`: (Optional) String used to separate multiple selected items. Default is ", ".
- `endprompt`: (Optional) Text that appears after the selected items.
- `items`: (Required) Array of items that can be selected for this field.
- `attributes`: (Optional) Object where keys are item names and values are arrays of attributes for that item.

## Example: Adding a "Visual Effects" Category

1. Create a new folder: `data/next/visual_effects/`
2. Create a JSON file in this folder, e.g., `effects.json`:

```json
{
  "preprompt": "with",
  "separator": " and ",
  "endprompt": "visual effects",
  "items": [
    "motion blur",
    "lens flare",
    "particle effects",
    "color grading",
    "depth of field"
  ],
  "attributes": {
    "motion blur": ["dynamic", "speed-enhancing", "cinematic"],
    "lens flare": ["bright", "atmospheric", "sci-fi-inspired"],
    "particle effects": ["intricate", "flowing", "ethereal"],
    "color grading": ["vibrant", "mood-setting", "stylized"],
    "depth of field": ["focused", "bokeh-rich", "photorealistic"]
  }
}
```

## Using Your New Category

After adding your new category and JSON file(s), the `APNextNode` function will automatically detect and include it as an optional input. Users can then select items from your new category when using the function for image generation prompts.

For example, using the "Visual Effects" category we just created:

- If a user selects "lens flare", the output might be: "with lens flare visual effects"
- If "attributes" is set to True and "lens flare" is selected, the output could be: "with lens flare (bright, atmospheric, sci-fi-inspired) visual effects"

Remember that the `APNextNode` function will handle the random selection and formatting based on the JSON structure you provide. This can greatly enhance the variety and specificity of prompts for AI image generation.

Here's a more professional version of the text, formatted as a README.md:

# ComfyUI Node Family

This new family of nodes for ComfyUI offers extensive flexibility and capabilities for prompt engineering and image generation workflows.

## Overview

![Node Family Overview](https://github.com/user-attachments/assets/89c23e6f-44f5-4d2f-bb37-abf8cbd797c4)

The system includes numerous nodes that can be chained together to create complex workflows:

![Node Chaining Example](https://github.com/user-attachments/assets/bf402844-ffdc-4dcf-bc6c-28d40e125011)

## Features

### GPT-4 Integration

Enhance prompts using the GPT-4 node:

![GPT-4 Node](https://github.com/user-attachments/assets/3cff0d18-a6e7-43c3-be5a-b2fb8964fa23)

### Local Ollama Support

Utilize local language models with the Ollama node:

![Ollama Node](https://github.com/user-attachments/assets/9d8f7eaa-07c2-48a2-bf67-37ebfbbfa4ba)

### Image-Based Prompt Generation

Create prompts based on images using various vision models:

![Image-Based Prompt Generation](https://github.com/user-attachments/assets/c9bbceaf-4a84-4d89-aecd-6026bafa1ab7)

### Dynamic Prompt Generation

Automatically incorporate LORA tokens using pre-defined prompts:

![Dynamic Prompt Generation](https://github.com/user-attachments/assets/1e69febe-2963-426c-b308-56766006b05e)

### Random Prompt Generator

Generate completely random prompts without the need for external language models:

![Random Prompt Generator](https://github.com/user-attachments/assets/3db6b94a-adca-4853-8530-5a95a79bceb7)

## Installation and Usage

1. Download the example workflow: [apntest.json](https://github.com/user-attachments/files/16830214/apntest.json)
2. To use GPT workflows, set your OpenAI API key in the environment:
   ```
   set OPENAI_API_KEY=sk-your-api-key-here
   ```
3. Run ComfyUI

## Custom Extensions

Add your own custom folders within `comfyui_dagthomas/data/next` with custom properties. These will be loaded in ComfyUI alongside the other nodes.

## Note

This project is currently in beta. Detailed documentation is in progress. Explore the various nodes and their capabilities to unlock the full potential of this ComfyUI extension.
