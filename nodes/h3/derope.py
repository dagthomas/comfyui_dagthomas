# APNext H3 De-Rope + Save Clip
#
# Fuses matlowai's Motion Lab de-rope pass into ONE per-clip node:
#   H3JerkOracle -> decode -> H3TimeSmear -> VAEEncode -> (H3AudioSmear ->
#   audio VAE) -> H3V2VInit -> H3InjectSchedule -> partial-denoise sampling ->
#   decode -> H3ExactRecover -> save to disk.
#
# The de-rope algorithm and its five stage nodes are from ComfyUI-MAINodes
# (https://github.com/matlowai/ComfyUI-MAINodes) - ALL credit to matlowai.
# This node calls those registered nodes rather than reimplementing them, so
# the pack must be installed. What the fusion adds is per-clip memory
# behaviour for music-video runs: a scene list maps through ONE node, so each
# clip's pass-1 frames, smeared frames and pass-2 frames are freed before the
# next clip starts, and only file-path strings accumulate. As ten separate
# nodes, every stage would cache every clip's frames and OOM long songs.

import logging

try:
    import torch
    import comfy.model_management
    import comfy.sample
    import comfy.samplers
    import comfy.utils
    import latent_preview
except ImportError:  # imported outside ComfyUI (widget checks) - node can't run there anyway
    pass

from ...utils.constants import CUSTOM_CATEGORY
from .clip_save import _FORMATS, save_frames

_SCHEDULERS = ["beta", "simple", "normal", "sgm_uniform", "karras", "exponential"]

try:
    _SAMPLERS = list(comfy.samplers.KSampler.SAMPLERS)
except Exception:
    _SAMPLERS = ["er_sde"]
if "er_sde" not in _SAMPLERS:
    _SAMPLERS.insert(0, "er_sde")

# H3V2VInit's plain-language audio presets; "invent freely" skips seeding
AUDIO_FOLLOW = [
    "follow the original performance (0.5)",
    "pin the original outright (0.0)",
    "follow loosely, re-render more (0.7)",
    "invent freely (original behaviour)",
]


def _mai(name):
    """A Motion Lab node class from the live ComfyUI registry."""
    import nodes as comfy_nodes
    cls = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(
            f"H3 De-Rope: node `{name}` is not installed. Install ComfyUI-MAINodes "
            "(https://github.com/matlowai/ComfyUI-MAINodes) into custom_nodes and "
            "restart ComfyUI."
        )
    return cls


def _decode_frames(vae, latent_samples):
    if latent_samples.is_nested:  # H3 AV latent: video stream first, same as core VAE Decode
        latent_samples = latent_samples.unbind()[0]
    images = vae.decode(latent_samples)
    if len(images.shape) == 5:  # combine video batches, same as core VAE Decode
        images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    return images


class H3DeRopeSave:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {
                    "tooltip": (
                        "The pass-1 sampler output (H3 AV latent). With a scene list "
                        "wired in, each clip is de-roped and saved before the next "
                        "one is touched."
                    ),
                }),
                "guider": ("GUIDER", {
                    "tooltip": "The SAME BasicGuider the pass-1 sampler used (same model, same conditioning).",
                }),
                "model": ("MODEL", {"tooltip": "The patched H3 model (for the injection schedule)."}),
                "vae": ("VAE", {"tooltip": "The MiniMax H3 video VAE."}),
                "filename_prefix": ("STRING", {
                    "default": "video/H3",
                    "tooltip": "Wire a writer's `project_name` output so every run gets its own folder.",
                }),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0}),
                "format": (_FORMATS, {"tooltip": "mp4/mkv use H.264, webm uses AV1."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "tooltip": "Noise seed for the de-rope regeneration pass."}),
                "steps": ("INT", {"default": 14, "min": 4, "max": 100,
                                  "tooltip": "Match the MAIN sampler's step count; `inject` decides how many actually run."}),
                "inject": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                     "tooltip": "How much of the denoise runs on the smeared init. 0.5 = faithful (metric best), 0.7 = balanced, 0.8 = loose."}),
                "scheduler": (_SCHEDULERS, {"default": "beta"}),
                "sampler_name": (_SAMPLERS, {"default": "er_sde"}),
                "q": ("FLOAT", {"default": 0.75, "min": 0.5, "max": 0.99, "step": 0.01,
                                "tooltip": "Jerk quantile that counts as 'too fast'. Raise toward 0.85 for tighter spans (cheaper), lower toward 0.7 to catch more."}),
                "d_max": ("INT", {"default": 4, "min": 2, "max": 8,
                                  "tooltip": "Peak hold count on the hottest frames; 4 is the measured sweet spot."}),
                "bridge": ("INT", {"default": 8, "min": 0, "max": 20,
                                   "tooltip": "Close hold-map valleys inside one burst (prevents mid-burst hiccups). 0 = off."}),
                "enabled": ("BOOLEAN", {"default": True,
                                        "tooltip": "Off = skip the de-rope entirely and save the pass-1 clip as-is."}),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": (
                        "This clip's slice of the song (writer `audio_segments`). Used "
                        "twice: stretched onto the held clock to seed pass 2's audio "
                        "rows (keeps lip-sync through held spans), and muxed into the "
                        "saved file."
                    ),
                }),
                "audio_vae": ("VAE", {"tooltip": "The MiniMax H3 audio VAE - needed to seed pass 2's audio rows."}),
                "audio_follow": (AUDIO_FOLLOW, {
                    "default": AUDIO_FOLLOW[0],
                    "tooltip": (
                        "How hard pass 2 follows the stretched song. 'follow the "
                        "original performance' keeps dialogue/lip-sync through held "
                        "spans; 'invent freely' skips audio seeding (the original "
                        "Motion Lab behaviour)."
                    ),
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "report")
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = f"{CUSTOM_CATEGORY}/H3"
    DESCRIPTION = (
        "One-node temporal de-rope + save (Motion Lab by matlowai, fused): finds "
        "where a clip's motion is too fast for H3, holds those frames, regenerates "
        "the clip v2v at partial denoise with the song stretched onto the held "
        "clock, recovers exact real time by dropping the held frames, and writes "
        "the mp4. Per-clip memory: an 18-scene song needs the RAM of one clip. "
        "A de-roped clip renders as ~2.4-3x its frames - budget time accordingly."
    )

    def run(self, samples, guider, model, vae, filename_prefix, fps, format,
            seed, steps, inject, scheduler, sampler_name, q, d_max, bridge,
            enabled, audio=None, audio_vae=None, audio_follow=AUDIO_FOLLOW[0]):
        frames1 = _decode_frames(vae, samples["samples"])
        n = int(frames1.shape[0])

        if not enabled:
            path = save_frames(frames1, audio, filename_prefix, fps, format)
            del frames1
            return (path, "de-rope disabled - pass-1 clip saved as-is")

        # 1. where is motion too fast? (per-token jerk on the clip's own latent)
        oracle = _mai("H3JerkOracle")().read(
            samples, n, q, d_max, True, preset="custom", bridge=bridge, fps=int(fps),
        )
        hold_map = oracle[0]

        # 2. hold the fast frames (adaptive smear onto the 17k+5 grid)
        smeared, hold_used, dilated_len, report = _mai("H3TimeSmear")().smear(
            frames1, 4, hold_map=hold_map, fps=int(fps),
        )
        del frames1

        # 3. v2v init: encode the smeared frames; seed the audio rows with the
        #    song stretched onto the SAME held clock so lips stay on the lyric
        init = {"samples": vae.encode(smeared[:, :, :, :3])}
        del smeared
        audio_latent = None
        seed_audio = audio_follow != "invent freely (original behaviour)"
        if seed_audio and audio is not None and audio_vae is None:
            logging.warning(
                "H3 De-Rope: audio is wired but audio_vae is not - pass 2 invents "
                "its own audio, held spans may come back with rushed lips. Wire the "
                "H3 audio VAE into audio_vae."
            )
        if seed_audio and audio is not None and audio_vae is not None:
            import torchaudio
            stretched = _mai("H3AudioSmear")().smear(audio, hold_used, fps=int(fps))[0]
            wav = stretched["waveform"]
            vae_sr = getattr(audio_vae, "audio_sample_rate", 44100)
            if int(stretched["sample_rate"]) != vae_sr:
                wav = torchaudio.functional.resample(wav, int(stretched["sample_rate"]), vae_sr)
            audio_latent = {"samples": audio_vae.encode(wav.movedim(1, -1))}
        if audio_latent is not None:
            latent = _mai("H3V2VInit")().build(init, audio_latent=audio_latent, audio_mode=audio_follow)[0]
        else:
            latent = _mai("H3V2VInit")().build(init)[0]

        # 4. regenerate on the truncated schedule (same guider as pass 1)
        sigmas = _mai("H3InjectSchedule")().sigmas(model, scheduler, steps, inject, preset="custom")[0]
        sampler = comfy.samplers.sampler_object(sampler_name)
        latent_image = comfy.sample.fix_empty_latent_channels(guider.model_patcher, latent["samples"])
        callback = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1)
        out = guider.sample(
            comfy.sample.prepare_noise(latent_image, seed, None), latent_image,
            sampler, sigmas, denoise_mask=latent.get("noise_mask"),
            callback=callback, disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=seed,
        )
        out = out.to(comfy.model_management.intermediate_device())
        del latent, latent_image

        # 5. back to real time (drop the held frames) and save
        frames2 = _decode_frames(vae, out)
        del out
        recovered = _mai("H3ExactRecover")().recover(frames2, hold_used)[0]
        del frames2
        path = save_frames(recovered, audio, filename_prefix, fps, format)
        del recovered
        logging.info(f"🪢 H3 De-Rope: {n} -> {dilated_len} -> {n} frames, saved {path}")
        return (path, report)
