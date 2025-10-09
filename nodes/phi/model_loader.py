# PhiModelLoader Node

from ...utils.constants import CUSTOM_CATEGORY
import torch


class PhiModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_version": (["Phi-3.5-vision-instruct"],),
                "image_crops": ([4, 16], {"default": 4}),
                "attention_mechanism": (['flash_attention_2', 'sdpa', 'eager'], {"default": 'eager'})
            },
        }

    RETURN_TYPES = ("PHI_MODEL_PIPELINE",)
    RETURN_NAMES = ("phi_pipeline",)
    FUNCTION = "load_phi_model"
    CATEGORY = "LLM/Phi"

    def load_phi_model(self, model_version, image_crops, attention_mechanism):
        model_id = f"microsoft/{model_version}"
        model_checkpoint = os.path.join(folder_paths.models_dir, 'LLM', os.path.basename(model_id))
        if not os.path.exists(model_checkpoint):
            from huggingface_hub import snapshot_download
            snapshot_download(repo_id=model_id, local_dir=model_checkpoint, local_dir_use_symlinks=False)
        
        phi_model = AutoModelForCausalLM.from_pretrained(
            model_checkpoint, 
            device_map="cuda", 
            torch_dtype="auto", 
            trust_remote_code=True,
            attn_implementation=attention_mechanism
        )
        phi_processor = AutoProcessor.from_pretrained(model_id, 
            trust_remote_code=True, 
            num_crops=image_crops
        )

        phi_pipeline = PhiModelPipeline()
        phi_pipeline.model = phi_model
        phi_pipeline.processor = phi_processor

        return (phi_pipeline,)
