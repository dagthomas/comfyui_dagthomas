# ComfyUI Plugin: Advanced Prompt Generation and Image Analysis
# comfyui_dagthomas

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

You can find an example workflow in /examples/flux/dagthomas_gpt4o-vision_prompt_generator_combiner.json

![image](https://github.com/user-attachments/assets/113bf0d2-3e19-4596-ae98-2f9a116193a2)

There is a toggle to create movie posters (08/04/24)
![ComfyUI_00161_](https://github.com/user-attachments/assets/262c7d5b-9770-492f-89b1-5dc6b62c9cc3)



### GPT4MiniNode

Generates text using OpenAI's GPT-4 model based on input text.

Features:
- Accepts text input and generates enhanced descriptions
- Supports custom base prompts
- Offers options for "happy talk" (detailed descriptions) or simple outputs
- Includes compression options to limit output length

You can find an example workflow in /examples/flux/dagthomas_morbuto_gpt-4o-mini_florence_prompt_combiner_flux.json

![image](https://github.com/user-attachments/assets/738b5588-b352-467a-8254-57fc74ecbe3a)


### OllamaNode

Generates text using custom Ollama based on input text.

Features:
- Accepts text input and generates enhanced descriptions
- Supports custom base prompts
- Offers options for "happy talk" (detailed descriptions) or simple outputs
- Includes compression options to limit output length

You can find an example workflow in /examples/flux/dagthomas_morbuto_ollama_florence_prompt_combiner_flux.json

![image](https://github.com/user-attachments/assets/bc0b4d00-2b1f-4989-9abf-6175ce347978)


### Pure Florence workflow: 

- You can also use a pure local Florence workflow without any of the others. The prompt will have some bloat, but works fine with Flux

You can find an example workflow in /examples/flux/dagthomas_florence_prompt_combiner_flux.json

![image](https://github.com/user-attachments/assets/73aaace6-da42-4416-a39d-df4e68cb6aec)



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




