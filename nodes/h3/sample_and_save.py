# APNext H3 Sampler + Save Clip - one node that samples a scene, decodes it and
# writes the clip, so the clips appear on disk ONE BY ONE.
#
# ComfyUI maps a list over each node before it moves to the next node: with a
# writer's 20 scenes wired in, SamplerCustomAdvanced renders all 20 latents
# first, and only then does Save Clip start decoding - nothing reaches disk
# until the last scene has sampled, and a crash at scene 19 loses everything.
# Chain Render avoids this by rendering scene by scene inside one node; this is
# the same idea for the ordinary sampler path: SamplerCustomAdvanced's inputs
# plus Save Clip's, and every list item is sampled, decoded, muxed with its
# audio slice and saved before the next one starts. Watch the output folder
# fill up while the run is still going.

import torch

from ...utils.constants import CUSTOM_CATEGORY
from .chain_render import _call, _node
from .clip_save import _FORMATS, save_frames


class H3SampleAndSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE", {"tooltip": "Random Noise (the seed)."}),
                "guider": ("GUIDER", {"tooltip": "Basic Guider (model + the scene's conditioning)."}),
                "sampler": ("SAMPLER", {"tooltip": "KSampler Select."}),
                "sigmas": ("SIGMAS", {"tooltip": "Basic Scheduler."}),
                "latent_image": ("LATENT", {
                    "tooltip": (
                        "The scene's starting latent - Reference To Video's LATENT, or the Song Masked AV "
                        "Context's. With a list of scenes wired in, this node runs once per scene."
                    ),
                }),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE - the clip is decoded here."}),
                "filename_prefix": ("STRING", {
                    "default": "video/H3",
                    "tooltip": (
                        "Where the clips go under the output directory - wire a writer's `project_name` "
                        "output here so every run lands in its own folder."
                    ),
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "H3 renders at 24 fps."}),
                "format": (_FORMATS, {"tooltip": "mp4/mkv use H.264, webm uses AV1."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "This clip's soundtrack - the writer's `audio_segments` list, so each scene is muxed "
                        "with its own slice of the song. Empty = silent clips."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("output", "file_path")
    OUTPUT_TOOLTIPS = (
        "The sampled latent, exactly what SamplerCustomAdvanced's `output` would be - for VAE Decode Audio, "
        "a latent upscaler, or anything else that used to hang off the sampler.",
        "The clip that was just written.",
    )
    OUTPUT_NODE = True
    FUNCTION = "render"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "SamplerCustomAdvanced + Save Clip in one node: every scene is sampled, decoded, muxed with its "
        "audio slice and written to disk BEFORE the next scene starts sampling - the clips appear one by "
        "one while the run is still going, and a crash late in the run keeps everything already saved. "
        "Drop it in where SamplerCustomAdvanced -> H3 Save Clip was; `output` still feeds whatever the "
        "sampler used to feed."
    )

    def render(self, noise, guider, sampler, sigmas, latent_image, vae, filename_prefix, fps, format, audio=None):
        sampled = _call(_node("SamplerCustomAdvanced"), noise=noise, guider=guider, sampler=sampler,
                        sigmas=sigmas, latent_image=latent_image)[0]
        latent = sampled["samples"]
        if getattr(latent, "is_nested", False):      # H3 AV latent: the video stream first, as core VAE Decode does
            latent = latent.unbind()[0]
        images = vae.decode(latent)
        if len(images.shape) == 5:                   # combine video batches, as core VAE Decode does
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        path = save_frames(images, audio, filename_prefix, fps, format)
        frames = int(images.shape[0])
        del images, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\U0001f3ac H3 Sampler + Save Clip | {frames} frames -> {path}")
        return (sampled, path)
