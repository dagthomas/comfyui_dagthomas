# PhiModelInference Node

from ...utils.constants import CUSTOM_CATEGORY
import torch


class PhiModelInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phi_pipeline": ("PHI_MODEL_PIPELINE",),
                "user_prompt": ("STRING", {"default": '', "multiline": True}),
                "input_images": ("IMAGE",),
                "generation_temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                "max_output_tokens": ("INT", {"default": 2048, "min": 100, "max": 10000, "step": 500}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "run_phi_inference"
    CATEGORY = "LLM/Phi"

    def run_phi_inference(self, phi_pipeline, user_prompt, input_images, generation_temperature, max_output_tokens):
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
                return ("Error: Failed to load model and processor. Please run the Phi Model Loader node again.",)

        phi_model = phi_pipeline.model
        phi_processor = phi_pipeline.processor
        
        image_placeholders = ""
        processed_images = []
        for index, image in enumerate(input_images, 1):
            image_tensor = torch.unsqueeze(image, 0)
            pil_image = tensor2pil(image_tensor).convert('RGB')
            processed_images.append(pil_image)
            image_placeholders += f"<|image_{index}|>\n"
        
        messages = [
            {"role": "user", "content": user_prompt + image_placeholders},
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
        
        # Instead of unloading, we'll keep the model and processor loaded
        # This might use more memory but will prevent the need to reload frequently
        # If memory is a concern, you might want to implement a more sophisticated
        # caching mechanism or adjust this approach based on your specific needs
        
        return (generated_text,)
    