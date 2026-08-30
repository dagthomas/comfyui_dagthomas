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
#
# Where the clip's sound comes from, in order:
#   * `audio` connected      - that soundtrack (the writer's `audio_segments`:
#                              the clip's slice of the original song);
#   * else `audio_vae`       - the audio half of the SAMPLED latent, decoded -
#                              what H3 itself produced in this pass (with a
#                              partly-frozen Masked Song Latent, the song as H3
#                              re-synthesised it). This is the setting for the
#                              Comfy H3 Sync Sound challenge: "the audio has to
#                              come out of the same H3 pass as the video";
#   * else                   - silent clips.

import torch

from ...utils.constants import CUSTOM_CATEGORY
from .chain_render import _call, _node
from .clip_save import _FORMATS, save_clip_wav, save_frames


def decode_sampled_audio(sampled, audio_vae):
    """
    The audio half of a sampled H3 AV latent, decoded with the audio VAE, as an
    AUDIO dict - the sound H3 generated in this pass. Goes through core's
    VAE Decode Audio when it is available (same loudness handling as the node
    in the graph), else decodes directly. Returns None for a latent with no
    audio stream.
    """
    latent = sampled["samples"]
    if not getattr(latent, "is_nested", False) or len(latent.unbind()) < 2:
        print("⚠️ H3 Sampler + Save Clip: audio_vae is connected but the sampled latent has no audio stream - silent clip")
        return None
    try:
        out = _call(_node("VAEDecodeAudio"), vae=audio_vae, samples=sampled)[0]
    except Exception:
        wave = audio_vae.decode(latent.unbind()[-1]).movedim(-1, 1)
        std = torch.std(wave, dim=[1, 2], keepdim=True) * 5.0
        std[std < 1.0] = 1.0
        wave = wave / std
        out = {"waveform": wave,
               "sample_rate": int(getattr(audio_vae, "audio_sample_rate_output",
                                          getattr(audio_vae, "audio_sample_rate", 32000)))}
    wave = out["waveform"]
    if wave.ndim == 2:
        wave = wave.unsqueeze(0)
    wave = wave[:1].detach().to("cpu").float()
    if wave.shape[1] == 1:
        wave = wave.repeat(1, 2, 1)
    seconds = wave.shape[-1] / float(out["sample_rate"])
    print(f"🔊 H3 Sampler + Save Clip | audio decoded from the sampled latent: {seconds:.2f}s at {int(out['sample_rate'])} Hz")
    return {"waveform": wave.contiguous(), "sample_rate": int(out["sample_rate"])}


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
                        "with its own slice of the song. Empty = the sampled latent's own audio if `audio_vae` "
                        "is connected, otherwise silent clips."
                    ),
                }),
                "audio_vae": ("VAE", {
                    "tooltip": (
                        "The MiniMax H3 audio VAE. With `audio` unplugged, the audio half of the SAMPLED latent "
                        "is decoded here and muxed into the clip - the sound H3 generated in this pass (the "
                        "Sync Sound challenge rule: audio from the same H3 pass as the video, nothing muxed "
                        "on afterwards). Ignored when `audio` is connected."
                    ),
                }),
                # appended last so saved workflows keep their widget positions
                "wav_sidecar": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Also write each clip's audio as a WAV next to its video file (same name). "
                        "For MANUAL assembly in an editor: AAC in mp4 leaves a 2-14 ms hole at every "
                        "butt-join (encoder priming/padding), while the WAVs are sample-exact - lay "
                        "them under the clips and the joins are gapless. H3 Stitch Clips does not "
                        "need them."
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
        "sampler used to feed. With `audio` unplugged and `audio_vae` connected, the clip gets the audio "
        "H3 generated (decoded from the sampled latent) - the Sync Sound challenge setting."
    )

    def render(self, noise, guider, sampler, sigmas, latent_image, vae, filename_prefix, fps, format, audio=None,
               wav_sidecar=True, audio_vae=None):
        sampled = _call(_node("SamplerCustomAdvanced"), noise=noise, guider=guider, sampler=sampler,
                        sigmas=sigmas, latent_image=latent_image)[0]
        latent = sampled["samples"]
        if getattr(latent, "is_nested", False):      # H3 AV latent: the video stream first, as core VAE Decode does
            latent = latent.unbind()[0]
        if audio is None and audio_vae is not None:
            audio = decode_sampled_audio(sampled, audio_vae)
        images = vae.decode(latent)
        if len(images.shape) == 5:                   # combine video batches, as core VAE Decode does
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        path = save_frames(images, audio, filename_prefix, fps, format)
        if wav_sidecar and audio is not None:
            save_clip_wav(audio, path)
        frames = int(images.shape[0])
        del images, latent
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"\U0001f3ac H3 Sampler + Save Clip | {frames} frames -> {path}")
        return (sampled, path)
