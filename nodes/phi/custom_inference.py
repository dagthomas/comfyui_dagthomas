# PhiCustomModelInference Node

from ...utils.constants import CUSTOM_CATEGORY
from PIL import Image
import numpy as np
import re
import torch


class PhiCustomModelInference:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phi_pipeline": ("PHI_MODEL_PIPELINE",),
                "images": ("IMAGE",),
                "custom_prompt": ("STRING", {"multiline": True, "default": ""}),
                "additive_prompt": ("STRING", {"multiline": True, "default": ""}),
                "dynamic_prompt": ("BOOLEAN", {"default": False}),
                "tag": ("STRING", {"default": "ohwx man"}),
                "sex": ("STRING", {"default": "male"}),
                "words": ("STRING", {"default": "100"}),
                "pronouns": ("STRING", {"default": "him, his"}),
                "fade_percentage": (
                    "FLOAT",
                    {"default": 15.0, "min": 0.1, "max": 50.0, "step": 0.1},
                ),
                "generation_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "IMAGE")
    RETURN_NAMES = ("output", "clip_l", "faded_image")
    FUNCTION = "run_phi_custom_inference"
    CATEGORY = "LLM/Phi"

    @staticmethod
    def tensor2pil(image):
        return Image.fromarray(
            np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        )

    @staticmethod
    def pil2tensor(image):
        return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

    def fade_images(self, images, fade_percentage=15.0):
        if len(images) < 2:
            return images[0] if images else None

        fade_width = int(images[0].width * (fade_percentage / 100))
        total_width = sum(img.width for img in images) - fade_width * (len(images) - 1)
        max_height = max(img.height for img in images)

        combined_image = Image.new("RGB", (total_width, max_height))

        x_offset = 0
        for i, img in enumerate(images):
            if i == 0:
                combined_image.paste(img, (0, 0))
                x_offset = img.width - fade_width
            else:
                for x in range(fade_width):
                    factor = x / fade_width
                    for y in range(max_height):
                        if y < images[i - 1].height and y < img.height:
                            pixel1 = images[i - 1].getpixel(
                                (images[i - 1].width - fade_width + x, y)
                            )
                            pixel2 = img.getpixel((x, y))
                            blended_pixel = tuple(
                                int(pixel1[c] * (1 - factor) + pixel2[c] * factor)
                                for c in range(3)
                            )
                            combined_image.putpixel((x_offset + x, y), blended_pixel)

                combined_image.paste(
                    img.crop((fade_width, 0, img.width, img.height)),
                    (x_offset + fade_width, 0),
                )
                x_offset += img.width - fade_width

        return combined_image

    def extract_first_two_sentences(self, text):
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    def run_phi_custom_inference(
        self,
        phi_pipeline,
        images,
        custom_prompt,
        additive_prompt,
        dynamic_prompt,
        tag,
        sex,
        words,
        pronouns,
        fade_percentage,
        generation_temperature,
        max_output_tokens
    ):
        try:
            if phi_pipeline.model is None or phi_pipeline.processor is None:
                print("Model or processor not loaded. Attempting to reload...")
                try:
                    model_id = "microsoft/phi-2"  # Adjust this if you're using a different model
                    model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
                    
                    phi_pipeline.model = AutoModelForCausalLM.from_pretrained(
                        model_checkpoint, 
                        device_map="cuda", 
                        torch_dtype="auto", 
                        trust_remote_code=True
                    )
                    phi_pipeline.processor = AutoProcessor.from_pretrained(model_id, 
                        trust_remote_code=True
                    )
                    print("Model and processor reloaded successfully.")
                except Exception as e:
                    print(f"Failed to reload model and processor: {str(e)}")
                    return ("Error: Failed to load model and processor. Please run the Phi Model Loader node again.", "", None)

            phi_model = phi_pipeline.model
            phi_processor = phi_pipeline.processor

            if not dynamic_prompt:
                full_prompt = custom_prompt if custom_prompt else "Analyze this image."
            else:
                custom_prompt = custom_prompt.replace("##TAG##", tag.lower())
                custom_prompt = custom_prompt.replace("##SEX##", sex)
                custom_prompt = custom_prompt.replace("##PRONOUNS##", pronouns)
                custom_prompt = custom_prompt.replace("##WORDS##", words)

                full_prompt = f"{additive_prompt} {custom_prompt}".strip() if additive_prompt else custom_prompt

            if len(images.shape) == 4:
                pil_images = [self.tensor2pil(img) for img in images]
            else:
                pil_images = [self.tensor2pil(images)]

            combined_image = self.fade_images(pil_images, fade_percentage)
            
            image_placeholders = ""
            processed_images = []
            for index, image in enumerate(pil_images, 1):
                processed_images.append(image.convert('RGB'))
                image_placeholders += f"<|image_{index}|>\n"
            
            messages = [
                {"role": "user", "content": full_prompt + image_placeholders},
            ]
            
            formatted_prompt = phi_processor.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            model_inputs = phi_processor(formatted_prompt, processed_images, return_tensors="pt").to("cuda:0") 
            
            do_sample = generation_temperature > 0
            generation_config = { 
                "max_new_tokens": max_output_tokens, 
                "do_sample": do_sample,
            }
            if do_sample:
                generation_config["temperature"] = generation_temperature
            
            generated_ids = phi_model.generate(
                **model_inputs, 
                eos_token_id=phi_processor.tokenizer.eos_token_id, 
                **generation_config
            )
            generated_ids = generated_ids[:, model_inputs['input_ids'].shape[1]:]
            generated_text = phi_processor.batch_decode(generated_ids, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False)[0]

            faded_image_tensor = self.pil2tensor(combined_image)

            return (
                generated_text,
                self.extract_first_two_sentences(generated_text),
                faded_image_tensor,
            )

        except Exception as e:
            print(f"An error occurred: {e}")
            error_message = f"Error occurred while processing the request: {str(e)}"
            error_image = Image.new("RGB", (512, 512), color="red")
            return (error_message, error_message[:100], self.pil2tensor(error_image))   
               